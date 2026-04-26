/*
 * Hologram Visualizer — C Renderer
 * Raspberry Pi 3B, 1024x600 HDMI, ARGB8888 framebuffer
 *
 * Optimizations:
 * - ARM NEON SIMD: processes 4 pixels per instruction
 * - Sine LUT: 1024-entry table, zero trig calls in render loop
 * - Distance² LUT: eliminates sqrt() from plasma calculation
 * - Double buffering: renders to RAM buffer, flips to /dev/fb0 atomically
 * - 4 pthreads: each handles 150 rows (600 / 4), one per core
 * - IPC: reads from /tmp/hologram.shm written by Python daemon
 *
 * Build:
 * gcc -O3 -march=armv8-a+simd -mtune=cortex-a53 -ffast-math \
 * -o visualizer visualizer_render.c -lpthread -lm
 *
 * Run:
 * fbset -g 1024 600 1024 1200 32   # double height for page flip, 32bpp
 * ./visualizer
 */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <pthread.h>
#include <arm_neon.h>

/* ── Display config ─────────────────────────────────────────────────────── */

/* * =========================================================================
 * EXTERNAL MONITOR FLAG
 * Currently set for a 1920x1200 external display.
 * WHEN MOVING BACK TO THE NATIVE 1024x600 SCREEN:
 * Change SCREEN_W to 1024
 * Change SCREEN_H to 600
 * =========================================================================
 */
#define SCREEN_W      1920
#define SCREEN_H      1200

#define DISPLAY_W     1024
#define DISPLAY_H     600
#define BYTES_PP      4          /* ARGB8888 */
#define FB_SIZE (SCREEN_W * SCREEN_H * BYTES_PP)

/* ── Shared memory layout (must match Python) ───────────────────────────── */
#define SHM_PATH          "/tmp/hologram.shm"
#define SHM_SIZE          (128 * 1024)
#define MAGIC_VAL         0xDEADBEEF

#define OFF_MAGIC         0
#define OFF_SEQ           4
#define OFF_MODE          8
#define OFF_SHOW_ALBUM    12
#define OFF_SHOW_BANDS    16
#define OFF_ALBUM_SIZE    20
#define OFF_BPM           28     /* OFF_NUM_LEDS (24) removed — no LEDs */
#define OFF_BANDS         32    /* 16 * float32 = 64 bytes */
#define OFF_PALETTE       96    /* 5 * 3 uint8 */
#define OFF_PLASMA_ABCD   156   /* 4 * float32 */
#define OFF_PLASMA_CXCY   172   /* 2 * float32 */
#define OFF_STALA_COLORS  180   /* 7 * float32 */
#define OFF_ALBUM_READY   208
#define OFF_ALBUM_DATA    212

/* ── LUT sizes ──────────────────────────────────────────────────────────── */
#define SIN_LUT_SIZE      1024
#define SIN_LUT_MASK      (SIN_LUT_SIZE - 1)
#define DIST2_LUT_W       (DISPLAY_W / 2 + 1)
#define DIST2_LUT_H       (DISPLAY_H / 2 + 1)

/* ── Bands / palette ────────────────────────────────────────────────────── */
#define NUM_BANDS         16
#define NUM_COLORS        5

/* ── Thread count ───────────────────────────────────────────────────────── */
#define NUM_THREADS       4
#define ROWS_PER_THREAD   (DISPLAY_H / NUM_THREADS)   /* 150 */

/* ── Album art ──────────────────────────────────────────────────────────── */
#define MAX_ALBUM_SIZE    180

/* ════════════════════════════════════════════════════════════════════════════
   Global data
   ════════════════════════════════════════════════════════════════════════════ */

/* LUTs */
static int16_t  sin_lut[SIN_LUT_SIZE];
static uint32_t dist2_lut[DIST2_LUT_H][DIST2_LUT_W];

/* Double buffer — two full frames in RAM */
static uint32_t buf0[DISPLAY_H * DISPLAY_W];
static uint32_t buf1[DISPLAY_H * DISPLAY_W];
static uint32_t *back_buf  = buf0;
static uint32_t *front_buf = buf1;

/* Framebuffer */
static int      fb_fd = -1;
static uint32_t *fb   = NULL;
static struct fb_var_screeninfo vinfo;

/* Shared memory */
static uint8_t *shm = NULL;

/* Current render parameters (copied from SHM once per frame) */
typedef struct {
    float    bands[NUM_BANDS];
    uint8_t  palette[NUM_COLORS][3];
    float    a, b, c, d, cx, cy;
    float    stala_colors[7];
    uint32_t mode;        /* 0=plasma, 1=stalagmites */
    uint32_t show_album;
    uint32_t show_bands;
    uint32_t album_size;
    uint32_t bpm;
    float    t;
    uint8_t  album_data[MAX_ALBUM_SIZE * MAX_ALBUM_SIZE * 3];
    int      album_ready;
} RenderParams;

static RenderParams params;

/* Thread sync */
typedef struct {
    int          thread_id;
    int          start_row;
    int          end_row;
    RenderParams *p;
    uint32_t     *buf;
} ThreadArgs;

static pthread_t         threads[NUM_THREADS];
static ThreadArgs        thread_args[NUM_THREADS];
static pthread_barrier_t frame_barrier;

/* ════════════════════════════════════════════════════════════════════════════
   LUT construction
   ════════════════════════════════════════════════════════════════════════════ */

static void build_sin_lut(void) {
    for (int i = 0; i < SIN_LUT_SIZE; i++)
        sin_lut[i] = (int16_t)(sinf(i * 2.0f * M_PI / SIN_LUT_SIZE) * 32767.0f);
}

static void build_dist2_lut(void) {
    for (int dy = 0; dy < DIST2_LUT_H; dy++)
        for (int dx = 0; dx < DIST2_LUT_W; dx++)
            dist2_lut[dy][dx] = (uint32_t)(dx * dx + dy * dy);
}

static inline float lut_sin(int idx) {
    return sin_lut[idx & SIN_LUT_MASK] * (1.0f / 32767.0f);
}

static inline float lut_dist(int dx, int dy) {
    int adx = dx < 0 ? -dx : dx;
    int ady = dy < 0 ? -dy : dy;
    if (adx >= DIST2_LUT_W) adx = DIST2_LUT_W - 1;
    if (ady >= DIST2_LUT_H) ady = DIST2_LUT_H - 1;
    /* max possible dist2 is (512^2 + 300^2) = 352144
     * divide so max output = 2*PI, meaning the circular wave
     * completes the same number of cycles regardless of screen size */
    return dist2_lut[ady][adx] * (2.0f * M_PI / 352144.0f);
}

/* ════════════════════════════════════════════════════════════════════════════
   Color helpers
   ════════════════════════════════════════════════════════════════════════════ */

static inline uint32_t rgb_to_argb8888(uint8_t r, uint8_t g, uint8_t b) {
    return (0xFF << 24) | (r << 16) | (g << 8) | b;
}

static inline uint32_t palette_lerp(const RenderParams *p, int idx1, int idx2, float blend) {
    float inv = 1.0f - blend;
    uint8_t r = (uint8_t)(p->palette[idx1][0] * inv + p->palette[idx2][0] * blend);
    uint8_t g = (uint8_t)(p->palette[idx1][1] * inv + p->palette[idx2][1] * blend);
    uint8_t b = (uint8_t)(p->palette[idx1][2] * inv + p->palette[idx2][2] * blend);
    return rgb_to_argb8888(r, g, b);
}

/* ════════════════════════════════════════════════════════════════════════════
   Plasma renderer — one horizontal chunk
   ════════════════════════════════════════════════════════════════════════════ */

static void render_plasma_chunk(const RenderParams *p, uint32_t *buf,
                                int start_row, int end_row) {
    const float a  = p->a;
    const float b  = p->b;
    const float c  = p->c;
    const float d  = p->d;
    const float t  = p->t;
    const float cx = p->cx;
    const float cy = p->cy;

    const float cos_tb = cosf(t * b);
    const float sin_tb = sinf(t * b);
    const float cos_td = cosf(t * d);
    const float sin_td = sinf(t * d);

    float bass      = (p->bands[0]+p->bands[1]+p->bands[2]+p->bands[3])    * (1.0f/4/255);
    float low_mids  = (p->bands[4]+p->bands[5]+p->bands[6]+p->bands[7])    * (1.0f/4/255);
    float high_mids = (p->bands[8]+p->bands[9]+p->bands[10]+p->bands[11])  * (1.0f/4/255);
    float highs     = (p->bands[12]+p->bands[13]+p->bands[14]+p->bands[15])* (1.0f/4/255);

    float fw[NUM_COLORS];
    fw[0] = 0.3f + bass      * 0.7f;
    fw[1] = 0.3f + low_mids  * 0.7f;
    fw[2] = 0.3f + high_mids * 0.7f;
    fw[3] = 0.3f + highs     * 0.7f;
    fw[4] = 0.3f + highs     * 0.5f;

    float fw_sum = fw[0]+fw[1]+fw[2]+fw[3]+fw[4];
    float scale  = (NUM_COLORS - 1) / fw_sum;
    float cum[NUM_COLORS + 1];
    cum[0] = 0.0f;
    for (int i = 0; i < NUM_COLORS; i++)
        cum[i+1] = cum[i] + fw[i] * scale;

    float inv_w = 2.0f * M_PI / DISPLAY_W;
    float inv_h = 2.0f * M_PI / DISPLAY_H;

    int cx_px = (int)(cx / (2.0f * M_PI) * DISPLAY_W);
    int cy_px = (int)(cy / (2.0f * M_PI) * DISPLAY_H);

    for (int y = start_row; y < end_row; y++) {
        float yy    = y * inv_h;
        float wave2 = lut_sin((int)(yy * a * SIN_LUT_SIZE / (2*M_PI))
                               + (int)(sin_tb * 2 * SIN_LUT_SIZE / (2*M_PI)));
        float rot_y = yy * sin_td;
        uint32_t *row = buf + y * DISPLAY_W;
        int x = 0;

#ifdef __ARM_NEON
        for (; x <= DISPLAY_W - 4; x += 4) {
            float xx0 = x     * inv_w;
            float xx1 = (x+1) * inv_w;
            float xx2 = (x+2) * inv_w;
            float xx3 = (x+3) * inv_w;

            #define LSIN(xx)  lut_sin((int)((xx) * a * SIN_LUT_SIZE / (2*M_PI)) \
                                    + (int)(cos_tb * 2 * SIN_LUT_SIZE / (2*M_PI)))
            #define LCIRC(xi) lut_sin((int)(lut_dist((xi)-cx_px, y-cy_px) \
                         * SIN_LUT_SIZE/(2*M_PI)) \
                         - (int)(t*c*SIN_LUT_SIZE/(2*M_PI)))
            #define LROT(xx)  lut_sin((int)(((xx) * cos_td + rot_y) * SIN_LUT_SIZE / (2*M_PI)))

            float v[4];
            v[0] = LSIN(xx0) + wave2 + LCIRC(x  ) * 0.2f + LROT(xx0);
            v[1] = LSIN(xx1) + wave2 + LCIRC(x+1) * 0.2f + LROT(xx1);
            v[2] = LSIN(xx2) + wave2 + LCIRC(x+2) * 0.2f + LROT(xx2);
            v[3] = LSIN(xx3) + wave2 + LCIRC(x+3) * 0.2f + LROT(xx3);

            #undef LSIN
            #undef LCIRC
            #undef LROT

            float32x4_t vv = vld1q_f32(v);
            vv = vmaxq_f32(vv, vdupq_n_f32(-3.0f));
            vv = vminq_f32(vv, vdupq_n_f32( 3.0f));
            vv = vaddq_f32(vv, vdupq_n_f32(3.0f));
            vv = vmulq_f32(vv, vdupq_n_f32(1.0f / 6.0f));
            vv = vmulq_f32(vv, vdupq_n_f32(cum[NUM_COLORS]));
            float norm[4];
            vst1q_f32(norm, vv);

            for (int k = 0; k < 4; k++) {
                float scaled = norm[k];
                int ci = 0;
                for (int pi = 0; pi < NUM_COLORS - 1; pi++)
                    if (scaled > cum[pi+1]) ci = pi + 1;
                ci = ci < NUM_COLORS - 1 ? ci : NUM_COLORS - 2;
                float blend = (scaled - cum[ci]) / (cum[ci+1] - cum[ci] + 1e-6f);
                blend = blend < 0 ? 0 : (blend > 1 ? 1 : blend);
                row[x + k] = palette_lerp(p, ci, ci + 1, blend);
            }
        }
#endif

        for (; x < DISPLAY_W; x++) {
            float xx    = x * inv_w;
            float wave1 = lut_sin((int)(xx * a * SIN_LUT_SIZE / (2*M_PI))
                                   + (int)(cos_tb * 2 * SIN_LUT_SIZE / (2*M_PI)));
            float dist  = lut_dist(x - cx_px, y - cy_px);
            float wave3 = lut_sin((int)(dist * SIN_LUT_SIZE/(2*M_PI))
                     - (int)(t * c * SIN_LUT_SIZE/(2*M_PI))) * 0.2f;
            float wave4 = lut_sin((int)((xx * cos_td + rot_y) * SIN_LUT_SIZE / (2*M_PI)));
            float v     = wave1 + wave2 + wave3 + wave4;
            v = (v + 3.0f) / 6.0f;
            float scaled = v * cum[NUM_COLORS];
            int ci = 0;
            for (int pi = 0; pi < NUM_COLORS - 1; pi++)
                if (scaled > cum[pi+1]) ci = pi + 1;
            ci = ci < NUM_COLORS - 1 ? ci : NUM_COLORS - 2;
            float blend = (scaled - cum[ci]) / (cum[ci+1] - cum[ci] + 1e-6f);
            blend = blend < 0 ? 0 : (blend > 1 ? 1 : blend);
            row[x] = palette_lerp(p, ci, ci + 1, blend);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Stalagmite renderer — one horizontal chunk
   ════════════════════════════════════════════════════════════════════════════ */

static void render_stalagmites_chunk(const RenderParams *p, uint32_t *buf,
                                     int start_row, int end_row) {
    const int num_stala = 7;
    const float t = p->t;
    uint8_t primary[3], secondary[3];

    for (int c = 0; c < 3; c++) {
        primary[c]   = p->palette[NUM_COLORS-1][c];
        secondary[c] = p->palette[NUM_COLORS-2][c];
    }

    for (int y = start_row; y < end_row; y++) {
        uint32_t *row = buf + y * DISPLAY_W;

        uint8_t bg_r = (uint8_t)(primary[0] * 0.09f);
        uint8_t bg_g = (uint8_t)(primary[1] * 0.09f);
        uint8_t bg_b = (uint8_t)(primary[2] * 0.09f);

        for (int x = 0; x < DISPLAY_W; x++) {
            float r_acc = bg_r, g_acc = bg_g, b_acc = bg_b;

            for (int i = 0; i < num_stala; i++) {
                int   center_x  = (int)(DISPLAY_W * (i + 0.5f) / num_stala);
                float stala_h   = DISPLAY_H * (0.35f + (i % 3) * 0.15f);
                float sigma     = 12.0f + (i % 3) * 5.0f;
                float breathing = sinf(t * 0.4f + i * 1.1f) * 0.08f + 0.92f;
                int   tip_y     = (int)(DISPLAY_H - stala_h);

                if (y < tip_y) continue;

                float height_ratio = (y - tip_y) / stala_h;
                if (height_ratio > 1.0f) height_ratio = 1.0f;
                float tip_ratio = 1.0f - height_ratio;

                int   dx     = x - center_x;
                float sigma2 = sigma * sigma * (1.0f - tip_ratio * 0.85f);
                float taper  = expf(-(float)(dx*dx) / (2.0f * sigma2));
                float mask   = powf(height_ratio, 0.4f) * taper;

                float bv = p->stala_colors[i];
                uint8_t cr = (uint8_t)(primary[0] * (1-bv) + secondary[0] * bv);
                uint8_t cg = (uint8_t)(primary[1] * (1-bv) + secondary[1] * bv);
                uint8_t cb = (uint8_t)(primary[2] * (1-bv) + secondary[2] * bv);

                r_acc += mask * cr * breathing;
                g_acc += mask * cg * breathing;
                b_acc += mask * cb * breathing;
            }

            uint8_t fr  = r_acc > 255 ? 255 : (uint8_t)r_acc;
            uint8_t fg  = g_acc > 255 ? 255 : (uint8_t)g_acc;
            uint8_t fb2 = b_acc > 255 ? 255 : (uint8_t)b_acc;
            row[x] = rgb_to_argb8888(fr, fg, fb2);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Album art overlay
   ════════════════════════════════════════════════════════════════════════════ */

static void overlay_album_art(const RenderParams *p, uint32_t *buf) {
    if (!p->album_ready) return;
    int sz  = (int)p->album_size;
    int x0  = (DISPLAY_W - sz) / 2;
    int y0  = (DISPLAY_H - sz) / 2;
    int r2  = (sz / 2) * (sz / 2);
    int cx2 = sz / 2;
    int cy2 = sz / 2;

    for (int ay = 0; ay < sz; ay++) {
        int gy = y0 + ay;
        if (gy < 0 || gy >= DISPLAY_H) continue;
        for (int ax = 0; ax < sz; ax++) {
            int gx = x0 + ax;
            if (gx < 0 || gx >= DISPLAY_W) continue;

            int dx = ax - cx2;
            int dy = ay - cy2;
            if (dx*dx + dy*dy > r2) continue;

            int     src = (ay * sz + ax) * 3;
            uint8_t r   = p->album_data[src];
            uint8_t g   = p->album_data[src+1];
            uint8_t b   = p->album_data[src+2];
            buf[gy * DISPLAY_W + gx] = rgb_to_argb8888(r, g, b);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Frequency bars overlay
   ════════════════════════════════════════════════════════════════════════════ */

static void draw_freq_bars(const RenderParams *p, uint32_t *buf) {
    int   center_y     = (int)(DISPLAY_H * 0.82f);
    int   max_half_h   = (int)(DISPLAY_H * 0.08f);
    float total_w      = DISPLAY_W * 0.80f;
    int   bar_w        = (int)(total_w / NUM_BANDS);
    int   gap          = 3;
    int   actual_bar_w = bar_w - gap > 2 ? bar_w - gap : 2;
    int   start_x      = (int)((DISPLAY_W - total_w) / 2);

    for (int i = 0; i < NUM_BANDS; i++) {
        int half = (int)(p->bands[i] / 255.0f * max_half_h);
        if (half < 2) half = 2;

        int x0 = start_x + i * bar_w;
        int x1 = x0 + actual_bar_w;
        int y0 = center_y - half;
        int y1 = center_y + half;

        if (x0 < 0) x0 = 0;
        if (x1 > DISPLAY_W) x1 = DISPLAY_W;
        if (y0 < 0) y0 = 0;
        if (y1 > DISPLAY_H) y1 = DISPLAY_H;

        uint32_t white = rgb_to_argb8888(255, 255, 255);
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                buf[y * DISPLAY_W + x] = white;
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Thread worker
   ════════════════════════════════════════════════════════════════════════════ */

static void *render_thread(void *arg) {
    ThreadArgs *ta = (ThreadArgs *)arg;

    while (1) {
        pthread_barrier_wait(&frame_barrier);

        if (ta->p->mode == 0)
            render_plasma_chunk(ta->p, ta->buf, ta->start_row, ta->end_row);
        else
            render_stalagmites_chunk(ta->p, ta->buf, ta->start_row, ta->end_row);

        pthread_barrier_wait(&frame_barrier);

        if (ta->thread_id == 0) {
            if (ta->p->show_bands)
                draw_freq_bars(ta->p, ta->buf);
            if (ta->p->show_album)
                overlay_album_art(ta->p, ta->buf);
        }

        pthread_barrier_wait(&frame_barrier);
    }
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════════
   SHM reader
   ════════════════════════════════════════════════════════════════════════════ */

static int read_shm(RenderParams *p) {
    uint32_t magic = *(uint32_t *)(shm + OFF_MAGIC);
    if (magic != MAGIC_VAL) return 0;

    memcpy(p->bands, shm + OFF_BANDS, NUM_BANDS * sizeof(float));

    for (int i = 0; i < NUM_COLORS; i++) {
        int base = OFF_PALETTE + i * 3;
        p->palette[i][0] = shm[base];
        p->palette[i][1] = shm[base+1];
        p->palette[i][2] = shm[base+2];
    }

    float *abcd = (float *)(shm + OFF_PLASMA_ABCD);
    p->a = abcd[0]; p->b = abcd[1]; p->c = abcd[2]; p->d = abcd[3];
    float *cxcy = (float *)(shm + OFF_PLASMA_CXCY);
    p->cx = cxcy[0]; p->cy = cxcy[1];

    memcpy(p->stala_colors, shm + OFF_STALA_COLORS, 7 * sizeof(float));

    p->mode       = *(uint32_t *)(shm + OFF_MODE);
    p->show_album = *(uint32_t *)(shm + OFF_SHOW_ALBUM);
    p->show_bands = *(uint32_t *)(shm + OFF_SHOW_BANDS);
    p->album_size = *(uint32_t *)(shm + OFF_ALBUM_SIZE);
    p->bpm        = *(uint32_t *)(shm + OFF_BPM);

    p->album_ready = (*(uint32_t *)(shm + OFF_ALBUM_READY)) ? 1 : 0;
    if (p->album_ready) {
        int sz = (int)p->album_size;
        if (sz < 10 || sz > MAX_ALBUM_SIZE) {
            p->album_ready = 0;
        } else {
            memcpy(p->album_data, shm + OFF_ALBUM_DATA, sz * sz * 3);
            if (!(*(uint32_t *)(shm + OFF_ALBUM_READY)))
                p->album_ready = 0;
    }
}  

    return 1;
}

/* ════════════════════════════════════════════════════════════════════════════
   Framebuffer setup
   ════════════════════════════════════════════════════════════════════════════ */

static int open_framebuffer(void) {
    fb_fd = open("/dev/fb0", O_RDWR);
    if (fb_fd < 0) { perror("open /dev/fb0"); return 0; }

    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo) < 0) {
        perror("FBIOGET_VSCREENINFO"); return 0;
    }

    vinfo.xres = SCREEN_W;
    vinfo.yres = SCREEN_H;
    vinfo.xres_virtual   = SCREEN_W;
    vinfo.yres_virtual   = SCREEN_H * 2;
    vinfo.bits_per_pixel = 32;

    if (ioctl(fb_fd, FBIOPUT_VSCREENINFO, &vinfo) < 0) {
        perror("FBIOPUT_VSCREENINFO"); return 0;
    }

    size_t map_size = SCREEN_W * SCREEN_H * 2 * BYTES_PP;
    fb = (uint32_t *)mmap(NULL, map_size,
                          PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (fb == MAP_FAILED) { perror("mmap fb"); return 0; }

    printf("Framebuffer: %dx%d @ %dbpp\n",
           vinfo.xres, vinfo.yres, vinfo.bits_per_pixel);
    return 1;
}

static void flip_buffer(uint32_t *rendered_buf) {
    for (int y = 0; y < DISPLAY_H; y++) {
        memcpy(fb + y * SCREEN_W,
               rendered_buf + y * DISPLAY_W,
               DISPLAY_W * BYTES_PP);
    }
}
/* ════════════════════════════════════════════════════════════════════════════
   Shared memory setup
   ════════════════════════════════════════════════════════════════════════════ */

static int open_shm_file(void) {
    int fd = open(SHM_PATH, O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "Cannot open SHM %s — is Python daemon running?\n", SHM_PATH);
        return 0;
    }
    shm = (uint8_t *)mmap(NULL, SHM_SIZE,
                          PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (shm == MAP_FAILED) { perror("mmap shm"); return 0; }
    printf("Shared memory mapped: %s (%d bytes)\n", SHM_PATH, SHM_SIZE);
    return 1;
}

/* ════════════════════════════════════════════════════════════════════════════
   Main
   ════════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("Hologram Visualizer C Renderer\n");
    printf("Display: %dx%d, %d threads\n", DISPLAY_W, DISPLAY_H, NUM_THREADS);

    build_sin_lut();
    build_dist2_lut();
    printf("LUTs built: sin[%d], dist2[%dx%d]\n",
           SIN_LUT_SIZE, DIST2_LUT_W, DIST2_LUT_H);

    if (!open_framebuffer()) return 1;

    printf("Waiting for Python daemon (SHM)...\n");
    while (!open_shm_file()) sleep(1);
    while (*(uint32_t *)(shm + OFF_MAGIC) != MAGIC_VAL) {
        printf("  waiting for Python to write initial state...\n");
        usleep(500000);
    }
    printf("Python daemon found — starting render loop\n");

    pthread_barrier_init(&frame_barrier, NULL, NUM_THREADS + 1);

    read_shm(&params);
    params.t = 0.0f;

    for (int i = 0; i < NUM_THREADS; i++) {
        thread_args[i].thread_id = i;
        thread_args[i].start_row = i * ROWS_PER_THREAD;
        thread_args[i].end_row   = (i == NUM_THREADS-1) ? DISPLAY_H
                                                         : (i+1) * ROWS_PER_THREAD;
        thread_args[i].p         = &params;
        thread_args[i].buf       = back_buf;
        pthread_create(&threads[i], NULL, render_thread, &thread_args[i]);
        printf("Thread %d: rows %d-%d\n",
               i, thread_args[i].start_row, thread_args[i].end_row);
    }

    struct timespec frame_start, frame_end;
    const long target_ns = 1000000000L / 30;

    while (1) {
        clock_gettime(CLOCK_MONOTONIC, &frame_start);

        read_shm(&params);

        for (int i = 0; i < NUM_THREADS; i++)
            thread_args[i].buf = back_buf;

        pthread_barrier_wait(&frame_barrier);   /* start render */
        pthread_barrier_wait(&frame_barrier);   /* chunks done  */
        pthread_barrier_wait(&frame_barrier);   /* overlays done */

        flip_buffer(back_buf);

        uint32_t *tmp = back_buf;
        back_buf  = front_buf;
        front_buf = tmp;

        float bpm_factor = params.bpm / 120.0f;
        params.t += bpm_factor * 0.07f;

        clock_gettime(CLOCK_MONOTONIC, &frame_end);
        long elapsed_ns = (frame_end.tv_sec  - frame_start.tv_sec)  * 1000000000L
                        + (frame_end.tv_nsec - frame_start.tv_nsec);
        long sleep_ns = target_ns - elapsed_ns;
        if (sleep_ns > 0) {
            struct timespec ts = { 0, sleep_ns };
            nanosleep(&ts, NULL);
        }
    }

    return 0;
}