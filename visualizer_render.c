/*
 * Hologram Visualizer — C Renderer
 * Raspberry Pi 3B, 1024x600 HDMI, ARGB8888 framebuffer
 *
 * Modes:
 *   0 = Minimal plasma  (full-screen plasma + small album art + freq bars)
 *   1 = Info display    (large album art left, track info right, wave bars bottom)
 *
 * Build:
 * gcc -O3 -march=armv8-a+simd -mtune=cortex-a53 -ffast-math \
 * -o visualizer_render visualizer_render.c -lpthread -lm
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
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif
#ifdef SDL2_FALLBACK
#include <SDL2/SDL.h>
#endif

/* ── Display config ──────────────────────────────────────────────────────── */
#define SCREEN_W      1920
#define SCREEN_H      1200
#define DISPLAY_W     1024
#define DISPLAY_H     600
#define BYTES_PP      4
#define FB_SIZE       (SCREEN_W * SCREEN_H * BYTES_PP)

/* ── Shared memory layout ────────────────────────────────────────────────── */
#define SHM_PATH          "/tmp/hologram.shm"
#define SHM_SIZE          (512 * 1024)
#define MAGIC_VAL         0xDEADBEEF

#define OFF_MAGIC         0
#define OFF_SEQ           4
#define OFF_MODE          8
#define OFF_SHOW_ALBUM    12
#define OFF_SHOW_BANDS    16
#define OFF_ALBUM_SIZE    20
#define OFF_BPM           28
#define OFF_BANDS         32     /* 16 * float32 = 64 bytes */
#define OFF_PALETTE       96     /* 5 * 3 uint8 = 15 bytes */
#define OFF_PLASMA_ABCD   156    /* 4 * float32 */
#define OFF_PLASMA_CXCY   172    /* 2 * float32 */
#define OFF_STALA_COLORS  180    /* 7 * float32 */
#define OFF_ALBUM_READY   208
#define OFF_ALBUM_DATA    212    /* 350*350*3 = 367500 bytes */

#define OFF_TRACK_TITLE   367712  /* 64 bytes */
#define OFF_TRACK_ARTIST  367776  /* 64 bytes */
#define OFF_TRACK_ALBUM_S 367840  /* 64 bytes */
#define OFF_TRACK_DUR_MS  367904  /* uint32 */
#define OFF_TRACK_POS_MS  367908  /* uint32 */
#define OFF_DISPLAY_MODE  367912  /* uint32: 0=plasma 1=info */
#define OFF_POS_UPDATED   367916  /* uint32 rolling counter */

/* ── LUT / render sizes ──────────────────────────────────────────────────── */
#define SIN_LUT_SIZE      1024
#define SIN_LUT_MASK      (SIN_LUT_SIZE - 1)
#define DIST2_LUT_W       (DISPLAY_W / 2 + 1)
#define DIST2_LUT_H       (DISPLAY_H / 2 + 1)
#define NUM_BANDS         16
#define NUM_COLORS        5
#define NUM_THREADS       4
#define ROWS_PER_THREAD   (DISPLAY_H / NUM_THREADS)
#define MAX_ALBUM_SIZE    350

/* ── Info mode layout ────────────────────────────────────────────────────── */
#define INFO_ART_SIZE     350
#define INFO_ART_X        20
#define INFO_ART_Y        ((DISPLAY_H - INFO_ART_SIZE) / 2)   /* 125 */
#define INFO_TEXT_X       400
#define INFO_TITLE_Y      140
#define INFO_ARTIST_Y     196
#define INFO_ALBUM_Y      224
#define INFO_PROG_X       400
#define INFO_PROG_Y       268
#define INFO_PROG_W       580
#define INFO_PROG_H       6
#define INFO_TIME_Y       282
#define WAVE_Y_TOP        490   /* wave bars start row */

/* ════════════════════════════════════════════════════════════════════════════
   Bitmap font — 8×8, ASCII 32-126
   ════════════════════════════════════════════════════════════════════════════ */
static const uint8_t font8x8[96][8] = {
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},  /* SPC */
    {0x18,0x3C,0x3C,0x18,0x18,0x00,0x18,0x00},  /* ! */
    {0x66,0x66,0x24,0x00,0x00,0x00,0x00,0x00},  /* " */
    {0x6C,0x6C,0xFE,0x6C,0xFE,0x6C,0x6C,0x00},  /* # */
    {0x18,0x3E,0x60,0x3C,0x06,0x7C,0x18,0x00},  /* $ */
    {0x00,0xC6,0xCC,0x18,0x30,0x66,0xC6,0x00},  /* % */
    {0x38,0x6C,0x38,0x76,0xDC,0xCC,0x76,0x00},  /* & */
    {0x18,0x18,0x30,0x00,0x00,0x00,0x00,0x00},  /* ' */
    {0x0C,0x18,0x30,0x30,0x30,0x18,0x0C,0x00},  /* ( */
    {0x30,0x18,0x0C,0x0C,0x0C,0x18,0x30,0x00},  /* ) */
    {0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00},  /* * */
    {0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00},  /* + */
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x30},  /* , */
    {0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00},  /* - */
    {0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00},  /* . */
    {0x06,0x0C,0x18,0x30,0x60,0xC0,0x80,0x00},  /* / */
    {0x3C,0x66,0x6E,0x76,0x66,0x66,0x3C,0x00},  /* 0 */
    {0x18,0x38,0x18,0x18,0x18,0x18,0x7E,0x00},  /* 1 */
    {0x3C,0x66,0x06,0x0C,0x30,0x60,0x7E,0x00},  /* 2 */
    {0x3C,0x66,0x06,0x1C,0x06,0x66,0x3C,0x00},  /* 3 */
    {0x0C,0x1C,0x3C,0x6C,0x7E,0x0C,0x0C,0x00},  /* 4 */
    {0x7E,0x60,0x7C,0x06,0x06,0x66,0x3C,0x00},  /* 5 */
    {0x1C,0x30,0x60,0x7C,0x66,0x66,0x3C,0x00},  /* 6 */
    {0x7E,0x06,0x0C,0x18,0x30,0x30,0x30,0x00},  /* 7 */
    {0x3C,0x66,0x66,0x3C,0x66,0x66,0x3C,0x00},  /* 8 */
    {0x3C,0x66,0x66,0x3E,0x06,0x0C,0x38,0x00},  /* 9 */
    {0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x00},  /* : */
    {0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x30},  /* ; */
    {0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00},  /* < */
    {0x00,0x00,0x7E,0x00,0x00,0x7E,0x00,0x00},  /* = */
    {0x60,0x30,0x18,0x0C,0x18,0x30,0x60,0x00},  /* > */
    {0x3C,0x66,0x06,0x0C,0x18,0x00,0x18,0x00},  /* ? */
    {0x3E,0x63,0x6F,0x69,0x6F,0x60,0x3E,0x00},  /* @ */
    {0x18,0x3C,0x66,0x7E,0x66,0x66,0x66,0x00},  /* A */
    {0x7C,0x66,0x66,0x7C,0x66,0x66,0x7C,0x00},  /* B */
    {0x3C,0x66,0x60,0x60,0x60,0x66,0x3C,0x00},  /* C */
    {0x78,0x6C,0x66,0x66,0x66,0x6C,0x78,0x00},  /* D */
    {0x7E,0x60,0x60,0x78,0x60,0x60,0x7E,0x00},  /* E */
    {0x7E,0x60,0x60,0x78,0x60,0x60,0x60,0x00},  /* F */
    {0x3C,0x66,0x60,0x6E,0x66,0x66,0x3C,0x00},  /* G */
    {0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x00},  /* H */
    {0x3C,0x18,0x18,0x18,0x18,0x18,0x3C,0x00},  /* I */
    {0x1E,0x0C,0x0C,0x0C,0x0C,0x6C,0x38,0x00},  /* J */
    {0x66,0x6C,0x78,0x70,0x78,0x6C,0x66,0x00},  /* K */
    {0x60,0x60,0x60,0x60,0x60,0x60,0x7E,0x00},  /* L */
    {0x63,0x77,0x7F,0x6B,0x63,0x63,0x63,0x00},  /* M */
    {0x66,0x76,0x7E,0x7E,0x6E,0x66,0x66,0x00},  /* N */
    {0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00},  /* O */
    {0x7C,0x66,0x66,0x7C,0x60,0x60,0x60,0x00},  /* P */
    {0x3C,0x66,0x66,0x66,0x66,0x3C,0x0E,0x00},  /* Q */
    {0x7C,0x66,0x66,0x7C,0x78,0x6C,0x66,0x00},  /* R */
    {0x3C,0x66,0x60,0x3C,0x06,0x66,0x3C,0x00},  /* S */
    {0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x00},  /* T */
    {0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x00},  /* U */
    {0x66,0x66,0x66,0x66,0x66,0x3C,0x18,0x00},  /* V */
    {0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00},  /* W */
    {0x66,0x66,0x3C,0x18,0x3C,0x66,0x66,0x00},  /* X */
    {0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x00},  /* Y */
    {0x7E,0x06,0x0C,0x18,0x30,0x60,0x7E,0x00},  /* Z */
    {0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00},  /* [ */
    {0xC0,0x60,0x30,0x18,0x0C,0x06,0x02,0x00},  /* \ */
    {0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00},  /* ] */
    {0x10,0x38,0x6C,0xC6,0x00,0x00,0x00,0x00},  /* ^ */
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0xFF},  /* _ */
    {0x30,0x18,0x0C,0x00,0x00,0x00,0x00,0x00},  /* ` */
    {0x00,0x00,0x3C,0x06,0x3E,0x66,0x3E,0x00},  /* a */
    {0x60,0x60,0x7C,0x66,0x66,0x66,0x7C,0x00},  /* b */
    {0x00,0x00,0x3C,0x60,0x60,0x60,0x3C,0x00},  /* c */
    {0x06,0x06,0x3E,0x66,0x66,0x66,0x3E,0x00},  /* d */
    {0x00,0x00,0x3C,0x66,0x7E,0x60,0x3C,0x00},  /* e */
    {0x1C,0x30,0x30,0x7C,0x30,0x30,0x30,0x00},  /* f */
    {0x00,0x00,0x3E,0x66,0x66,0x3E,0x06,0x7C},  /* g */
    {0x60,0x60,0x7C,0x66,0x66,0x66,0x66,0x00},  /* h */
    {0x18,0x00,0x38,0x18,0x18,0x18,0x3C,0x00},  /* i */
    {0x06,0x00,0x06,0x06,0x06,0x66,0x66,0x3C},  /* j */
    {0x60,0x60,0x66,0x6C,0x78,0x6C,0x66,0x00},  /* k */
    {0x38,0x18,0x18,0x18,0x18,0x18,0x3C,0x00},  /* l */
    {0x00,0x00,0x66,0x7F,0x7F,0x6B,0x63,0x00},  /* m */
    {0x00,0x00,0x7C,0x66,0x66,0x66,0x66,0x00},  /* n */
    {0x00,0x00,0x3C,0x66,0x66,0x66,0x3C,0x00},  /* o */
    {0x00,0x00,0x7C,0x66,0x66,0x7C,0x60,0x60},  /* p */
    {0x00,0x00,0x3E,0x66,0x66,0x3E,0x06,0x06},  /* q */
    {0x00,0x00,0x7C,0x66,0x60,0x60,0x60,0x00},  /* r */
    {0x00,0x00,0x3E,0x60,0x3C,0x06,0x7C,0x00},  /* s */
    {0x18,0x18,0x7E,0x18,0x18,0x18,0x0E,0x00},  /* t */
    {0x00,0x00,0x66,0x66,0x66,0x66,0x3E,0x00},  /* u */
    {0x00,0x00,0x66,0x66,0x66,0x3C,0x18,0x00},  /* v */
    {0x00,0x00,0x63,0x6B,0x7F,0x7F,0x36,0x00},  /* w */
    {0x00,0x00,0x66,0x3C,0x18,0x3C,0x66,0x00},  /* x */
    {0x00,0x00,0x66,0x66,0x66,0x3E,0x06,0x7C},  /* y */
    {0x00,0x00,0x7E,0x0C,0x18,0x30,0x7E,0x00},  /* z */
    {0x0E,0x18,0x18,0x70,0x18,0x18,0x0E,0x00},  /* { */
    {0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00},  /* | */
    {0x70,0x18,0x18,0x0E,0x18,0x18,0x70,0x00},  /* } */
    {0x76,0xDC,0x00,0x00,0x00,0x00,0x00,0x00},  /* ~ */
};

/* ════════════════════════════════════════════════════════════════════════════
   Global data
   ════════════════════════════════════════════════════════════════════════════ */

static int16_t  sin_lut[SIN_LUT_SIZE];
static uint32_t dist2_lut[DIST2_LUT_H][DIST2_LUT_W];

static uint32_t buf0[DISPLAY_H * DISPLAY_W];
static uint32_t buf1[DISPLAY_H * DISPLAY_W];
static uint32_t *back_buf  = buf0;
static uint32_t *front_buf = buf1;

/* Stable album art cache — only updated when seq changes */
static uint8_t  stable_album[MAX_ALBUM_SIZE * MAX_ALBUM_SIZE * 3];
static int      stable_album_ready = 0;
static uint32_t stable_album_seq   = 0xFFFFFFFF;
static int      stable_album_size  = 0;

static int      fb_fd = -1;
static uint32_t *fb   = NULL;
#ifdef SDL2_FALLBACK
static SDL_Window   *sdl_window   = NULL;
static SDL_Renderer *sdl_renderer = NULL;
static SDL_Texture  *sdl_texture  = NULL;
#endif
static struct fb_var_screeninfo vinfo;
static volatile uint8_t *shm = NULL;

typedef struct {
    float    bands[NUM_BANDS];
    uint8_t  palette[NUM_COLORS][3];
    float    a, b, c, d, cx, cy;
    float    stala_colors[7];
    uint32_t mode;
    uint32_t show_album;
    uint32_t show_bands;
    uint32_t album_size;
    uint32_t bpm;
    float    t;
    uint8_t  album_data[MAX_ALBUM_SIZE * MAX_ALBUM_SIZE * 3];
    int      album_ready;
    /* info mode fields */
    uint32_t display_mode;   /* 0=plasma, 1=info */
    char     track_title[64];
    char     track_artist[64];
    char     track_album[64];
    uint32_t track_dur_ms;
    uint32_t track_pos_ms;
} RenderParams;

static RenderParams params_buf[2];
static int          params_idx = 0;

typedef struct {
    int       thread_id;
    int       start_row;
    int       end_row;
    int       p_idx;
    uint32_t *buf;
} ThreadArgs;

static pthread_t         threads[NUM_THREADS];
static ThreadArgs        thread_args[NUM_THREADS];
static pthread_barrier_t frame_barrier;

/* ════════════════════════════════════════════════════════════════════════════
   LUTs
   ════════════════════════════════════════════════════════════════════════════ */

static void build_sin_lut(void) {
    for (int i = 0; i < SIN_LUT_SIZE; i++)
        sin_lut[i] = (int16_t)(sinf(i * 2.0f * M_PI / SIN_LUT_SIZE) * 32767.0f);
}

static void build_dist2_lut(void) {
    for (int dy = 0; dy < DIST2_LUT_H; dy++)
        for (int dx = 0; dx < DIST2_LUT_W; dx++)
            dist2_lut[dy][dx] = (uint32_t)(dx*dx + dy*dy);
}

static inline float lut_sin(int idx) {
    return sin_lut[idx & SIN_LUT_MASK] * (1.0f / 32767.0f);
}

static inline float lut_dist(int dx, int dy) {
    int adx = dx < 0 ? -dx : dx;
    int ady = dy < 0 ? -dy : dy;
    if (adx >= DIST2_LUT_W) adx = DIST2_LUT_W - 1;
    if (ady >= DIST2_LUT_H) ady = DIST2_LUT_H - 1;
    return dist2_lut[ady][adx] * (2.0f * M_PI / 352144.0f);
}

/* ════════════════════════════════════════════════════════════════════════════
   Color helpers
   ════════════════════════════════════════════════════════════════════════════ */

static inline uint32_t rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return ((uint32_t)a << 24) | ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;
}

static inline uint32_t rgb(uint8_t r, uint8_t g, uint8_t b) {
    return rgba(r, g, b, 0xFF);
}

static inline uint32_t palette_lerp(const RenderParams *p, int i1, int i2, float blend) {
    float inv = 1.0f - blend;
    return rgb(
        (uint8_t)(p->palette[i1][0]*inv + p->palette[i2][0]*blend),
        (uint8_t)(p->palette[i1][1]*inv + p->palette[i2][1]*blend),
        (uint8_t)(p->palette[i1][2]*inv + p->palette[i2][2]*blend));
}

/* ════════════════════════════════════════════════════════════════════════════
   Bitmap font renderer
   ════════════════════════════════════════════════════════════════════════════ */

static void draw_char(uint32_t *buf, int x, int y, char c, int scale,
                      uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) {
    if (c < 32 || c > 126) c = '?';
    const uint8_t *glyph = font8x8[(int)(c - 32)];
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            if (!(glyph[row] & (0x80 >> col))) continue;
            for (int sy = 0; sy < scale; sy++) {
                int py = y + row*scale + sy;
                if (py < 0 || py >= DISPLAY_H) continue;
                for (int sx = 0; sx < scale; sx++) {
                    int px = x + col*scale + sx;
                    if (px < 0 || px >= DISPLAY_W) continue;
                    if (alpha == 255) {
                        buf[py * DISPLAY_W + px] = rgb(r, g, b);
                    } else {
                        /* alpha blend over existing pixel */
                        uint32_t bg = buf[py * DISPLAY_W + px];
                        uint8_t br = (bg >> 16) & 0xFF;
                        uint8_t bg2 = (bg >> 8) & 0xFF;
                        uint8_t bb = bg & 0xFF;
                        float a = alpha / 255.0f;
                        float ia = 1.0f - a;
                        buf[py * DISPLAY_W + px] = rgb(
                            (uint8_t)(r*a + br*ia),
                            (uint8_t)(g*a + bg2*ia),
                            (uint8_t)(b*a + bb*ia));
                    }
                }
            }
        }
    }
}

static void draw_text(uint32_t *buf, int x, int y, const char *text, int scale,
                      uint8_t r, uint8_t g, uint8_t b, uint8_t alpha) {
    int cx = x;
    for (int i = 0; text[i]; i++) {
        draw_char(buf, cx, y, text[i], scale, r, g, b, alpha);
        cx += 8 * scale + scale;
    }
}

/* truncate text to fit max_chars, append "..." if truncated */
static void truncate_text(const char *src, char *dst, int max_chars) {
    int len = 0;
    while (src[len]) len++;
    if (len <= max_chars) {
        memcpy(dst, src, len + 1);
    } else {
        memcpy(dst, src, max_chars - 3);
        dst[max_chars-3] = '.';
        dst[max_chars-2] = '.';
        dst[max_chars-1] = '.';
        dst[max_chars]   = '\0';
    }
}

static void format_time(uint32_t ms, char *out) {
    uint32_t secs = ms / 1000;
    uint32_t m = secs / 60;
    uint32_t s = secs % 60;
    out[0] = '0' + (m / 10);
    out[1] = '0' + (m % 10);
    out[2] = ':';
    out[3] = '0' + (s / 10);
    out[4] = '0' + (s % 10);
    out[5] = '\0';
}

/* ════════════════════════════════════════════════════════════════════════════
   Plasma renderer
   ════════════════════════════════════════════════════════════════════════════ */

static void render_plasma_chunk(const RenderParams *p, uint32_t *buf,
                                int start_row, int end_row, float dim) {
    const float a  = p->a, b = p->b, c = p->c, d = p->d, t = p->t;
    const float cx = p->cx, cy = p->cy;
    const float cos_tb = cosf(t*b), sin_tb = sinf(t*b);
    const float cos_td = cosf(t*d), sin_td = sinf(t*d);

    float bass      = (p->bands[0]+p->bands[1]+p->bands[2]+p->bands[3])    *(1.0f/4/255);
    float low_mids  = (p->bands[4]+p->bands[5]+p->bands[6]+p->bands[7])    *(1.0f/4/255);
    float high_mids = (p->bands[8]+p->bands[9]+p->bands[10]+p->bands[11])  *(1.0f/4/255);
    float highs     = (p->bands[12]+p->bands[13]+p->bands[14]+p->bands[15])*(1.0f/4/255);

    float fw[NUM_COLORS];
    fw[0]=0.3f+bass*0.7f; fw[1]=0.3f+low_mids*0.7f;
    fw[2]=0.3f+high_mids*0.7f; fw[3]=0.3f+highs*0.7f; fw[4]=0.3f+highs*0.5f;
    float fw_sum=fw[0]+fw[1]+fw[2]+fw[3]+fw[4];
    float scale=(NUM_COLORS-1)/fw_sum;
    float cum[NUM_COLORS+1]; cum[0]=0.0f;
    for (int i=0;i<NUM_COLORS;i++) cum[i+1]=cum[i]+fw[i]*scale;

    float inv_w=2.0f*M_PI/DISPLAY_W, inv_h=2.0f*M_PI/DISPLAY_H;
    int cx_px=(int)(cx/(2.0f*M_PI)*DISPLAY_W);
    int cy_px=(int)(cy/(2.0f*M_PI)*DISPLAY_H);

    for (int y=start_row; y<end_row; y++) {
        float yy=y*inv_h;
        float wave2=lut_sin((int)(yy*a*SIN_LUT_SIZE/(2*M_PI))
                            +(int)(sin_tb*2*SIN_LUT_SIZE/(2*M_PI)));
        float rot_y=yy*sin_td;
        uint32_t *row=buf+y*DISPLAY_W;

        for (int x=0;x<DISPLAY_W;x++) {
            float xx=x*inv_w;
            float w1=lut_sin((int)(xx*a*SIN_LUT_SIZE/(2*M_PI))+(int)(cos_tb*2*SIN_LUT_SIZE/(2*M_PI)));
            float dist=lut_dist(x-cx_px,y-cy_px);
            float w3=lut_sin((int)(dist*SIN_LUT_SIZE/(2*M_PI))-(int)(t*c*SIN_LUT_SIZE/(2*M_PI)))*0.2f;
            float w4=lut_sin((int)((xx*cos_td+rot_y)*SIN_LUT_SIZE/(2*M_PI)));
            float v=w1+wave2+w3+w4;
            float norm = (v + 3.0f) / 6.0f;
            if (norm < 0.0f) norm = 0.0f;
            if (norm > 1.0f) norm = 1.0f;
            /* plasma mode: darken midtones via fast quadratic (no powf) */
            if (dim > 0.9f) norm = norm * (2.0f - norm) * 0.6f;
            float scaled = norm * cum[NUM_COLORS];
            int ci=0;
            for (int pi=0;pi<NUM_COLORS-1;pi++) if (scaled>cum[pi+1]) ci=pi+1;
            ci = ci<NUM_COLORS-1 ? ci : NUM_COLORS-2;
            float blend=(scaled-cum[ci])/(cum[ci+1]-cum[ci]+1e-6f);
            blend=blend<0?0:(blend>1?1:blend);
            float inv_b=1.0f-blend;
            uint8_t r2=(uint8_t)((p->palette[ci][0]*inv_b+p->palette[ci+1][0]*blend)*dim);
            uint8_t g2=(uint8_t)((p->palette[ci][1]*inv_b+p->palette[ci+1][1]*blend)*dim);
            uint8_t b2=(uint8_t)((p->palette[ci][2]*inv_b+p->palette[ci+1][2]*blend)*dim);
            row[x]=rgb(r2,g2,b2);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Stalagmite renderer
   ════════════════════════════════════════════════════════════════════════════ */

static void render_stalagmites_chunk(const RenderParams *p, uint32_t *buf,
                                     int start_row, int end_row) {
    const float t=p->t;
    uint8_t primary[3], secondary[3];
    for (int c=0;c<3;c++) {
        primary[c]=p->palette[NUM_COLORS-1][c];
        secondary[c]=p->palette[NUM_COLORS-2][c];
    }
    for (int y=start_row;y<end_row;y++) {
        uint32_t *row=buf+y*DISPLAY_W;
        uint8_t bg_r=(uint8_t)(primary[0]*0.09f);
        uint8_t bg_g=(uint8_t)(primary[1]*0.09f);
        uint8_t bg_b=(uint8_t)(primary[2]*0.09f);
        for (int x=0;x<DISPLAY_W;x++) {
            float r_acc=bg_r,g_acc=bg_g,b_acc=bg_b;
            for (int i=0;i<7;i++) {
                int cx2=(int)(DISPLAY_W*(i+0.5f)/7);
                float sh=DISPLAY_H*(0.35f+(i%3)*0.15f);
                float sigma=12.0f+(i%3)*5.0f;
                float breathing=sinf(t*0.4f+i*1.1f)*0.08f+0.92f;
                int ty=(int)(DISPLAY_H-sh);
                if (y<ty) continue;
                float hr=(y-ty)/sh; if (hr>1.0f) hr=1.0f;
                float tr=1.0f-hr;
                int dx=x-cx2;
                float sig2=sigma*sigma*(1.0f-tr*0.85f);
                float taper=expf(-(float)(dx*dx)/(2.0f*sig2));
                float mask=powf(hr,0.4f)*taper;
                float bv=p->stala_colors[i];
                r_acc+=mask*((uint8_t)(primary[0]*(1-bv)+secondary[0]*bv))*breathing;
                g_acc+=mask*((uint8_t)(primary[1]*(1-bv)+secondary[1]*bv))*breathing;
                b_acc+=mask*((uint8_t)(primary[2]*(1-bv)+secondary[2]*bv))*breathing;
            }
            row[x]=rgb(r_acc>255?255:(uint8_t)r_acc,
                       g_acc>255?255:(uint8_t)g_acc,
                       b_acc>255?255:(uint8_t)b_acc);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Album art overlay (circular, centered — minimal mode)
   ════════════════════════════════════════════════════════════════════════════ */

static void overlay_album_art_centered(const RenderParams *p, uint32_t *buf) {
    if (!p->album_ready) return;
    int src_sz=(int)p->album_size;
    if (src_sz<10||src_sz>MAX_ALBUM_SIZE) return;
    /* cap to 160px so it stays above freq bars */
    int sz = src_sz > 160 ? 160 : src_sz;
    int x0=(DISPLAY_W-sz)/2, y0=(DISPLAY_H/2 - sz/2 - 40);
    int r2=(sz/2)*(sz/2), cx2=sz/2, cy2=sz/2;
    for (int ay=0;ay<sz;ay++) {
        int gy=y0+ay; if (gy<0||gy>=DISPLAY_H) continue;
        for (int ax=0;ax<sz;ax++) {
            int gx=x0+ax; if (gx<0||gx>=DISPLAY_W) continue;
            int dx=ax-cx2, dy=ay-cy2;
            if (dx*dx+dy*dy>r2) continue;
            int ssx=(ax*src_sz)/sz, ssy=(ay*src_sz)/sz;
            int src=(ssy*src_sz+ssx)*3;
            buf[gy*DISPLAY_W+gx]=rgb(p->album_data[src],
                                     p->album_data[src+1],
                                     p->album_data[src+2]);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Album art overlay — info mode (left-aligned square with rounded corners)
   ════════════════════════════════════════════════════════════════════════════ */

static void overlay_album_art_info(const RenderParams *p, uint32_t *buf) {
    if (!p->album_ready) return;
    int sz=INFO_ART_SIZE;
    if (stable_album_size < 10) return;
    /* scale from stable_album_size to INFO_ART_SIZE */
    int src_sz = stable_album_size;
    int x0=INFO_ART_X, y0=INFO_ART_Y;
    int corner=12; /* rounded corner radius */
    for (int ay=0;ay<sz;ay++) {
        int gy=y0+ay; if (gy<0||gy>=DISPLAY_H) continue;
        for (int ax=0;ax<sz;ax++) {
            int gx=x0+ax; if (gx<0||gx>=DISPLAY_W) continue;
            /* rounded corners */
            int cdx=0,cdy=0;
            if (ax<corner && ay<corner)         { cdx=corner-ax; cdy=corner-ay; }
            else if (ax>=sz-corner && ay<corner) { cdx=ax-(sz-corner-1); cdy=corner-ay; }
            else if (ax<corner && ay>=sz-corner) { cdx=corner-ax; cdy=ay-(sz-corner-1); }
            else if (ax>=sz-corner && ay>=sz-corner){ cdx=ax-(sz-corner-1); cdy=ay-(sz-corner-1); }
            if (cdx*cdx+cdy*cdy > corner*corner) continue;
            /* nearest-neighbour scale */
            int sx=(ax*src_sz)/sz, sy=(ay*src_sz)/sz;
            int src=(sy*src_sz+sx)*3;
            buf[gy*DISPLAY_W+gx]=rgb(stable_album[src],
                                     stable_album[src+1],
                                     stable_album[src+2]);
        }
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Wave frequency bars (info mode — sine envelope, filled solid)
   ════════════════════════════════════════════════════════════════════════════ */

static void draw_wave_bars(const RenderParams *p, uint32_t *buf) {
    /* Build envelope: for each x pixel, interpolate between band peaks
     * using a smooth curve. Bands are evenly spaced across DISPLAY_W. */
    const int wave_h = DISPLAY_H - WAVE_Y_TOP;   /* 110 pixels */
    const float band_spacing = (float)DISPLAY_W / NUM_BANDS;

    /* band peak heights in pixels */
    float peak[NUM_BANDS];
    for (int i=0;i<NUM_BANDS;i++)
        peak[i] = (p->bands[i] / 255.0f) * (wave_h * 0.85f);

    /* palette accent color for fill — use brightest palette color */
    uint8_t wr = p->palette[NUM_COLORS-1][0];
    uint8_t wg = p->palette[NUM_COLORS-1][1];
    uint8_t wb = p->palette[NUM_COLORS-1][2];
    /* ensure minimum brightness */
    if (wr<60&&wg<60&&wb<60) { wr=120; wg=200; wb=255; }

    for (int x=0;x<DISPLAY_W;x++) {
        /* find which two bands this x is between, cosine interpolate */
        float band_pos = (float)x / band_spacing - 0.5f;
        int b0 = (int)band_pos; if (b0<0) b0=0; if (b0>=NUM_BANDS-1) b0=NUM_BANDS-2;
        int b1 = b0+1;
        float t2 = band_pos - b0;
        /* cosine interpolation */
        float mu = (1.0f - cosf(t2 * M_PI)) * 0.5f;
        float h = peak[b0]*(1.0f-mu) + peak[b1]*mu;

        int top_y = DISPLAY_H - 1 - (int)h;
        if (top_y < WAVE_Y_TOP) top_y = WAVE_Y_TOP;

        for (int y=top_y; y<DISPLAY_H; y++) {
            float fade = 1.0f - (float)(y - top_y) / (float)(DISPLAY_H - top_y + 1);
            fade = fade * 0.75f + 0.25f;   /* floor at 25% so bottom stays visible */
            buf[y*DISPLAY_W+x] = rgb(
                (uint8_t)(wr*fade),
                (uint8_t)(wg*fade),
                (uint8_t)(wb*fade));
        }
    }

    /* draw the outline on top — brighter */
    for (int x=1;x<DISPLAY_W-1;x++) {
        float band_pos = (float)x / band_spacing - 0.5f;
        int b0 = (int)band_pos; if (b0<0) b0=0; if (b0>=NUM_BANDS-1) b0=NUM_BANDS-2;
        int b1 = b0+1;
        float t2 = band_pos - b0;
        float mu = (1.0f - cosf(t2 * M_PI)) * 0.5f;
        float h = peak[b0]*(1.0f-mu) + peak[b1]*mu;
        int top_y = DISPLAY_H - 1 - (int)h;
        if (top_y < WAVE_Y_TOP) top_y = WAVE_Y_TOP;
        if (top_y >= DISPLAY_H) top_y = DISPLAY_H-1;
        buf[top_y*DISPLAY_W+x] = rgb(255,255,255);
        if (top_y > 0)
            buf[(top_y-1)*DISPLAY_W+x] = rgb(200,220,255);
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Minimal freq bars (plasma mode)
   ════════════════════════════════════════════════════════════════════════════ */

static void draw_freq_bars(const RenderParams *p, uint32_t *buf) {
    int cy=(int)(DISPLAY_H*0.82f);
    int mhh=(int)(DISPLAY_H*0.08f);
    float tw=DISPLAY_W*0.80f;
    int bw=(int)(tw/NUM_BANDS), gap=3;
    int abw=bw-gap>2?bw-gap:2;
    int sx=(int)((DISPLAY_W-tw)/2);
    for (int i=0;i<NUM_BANDS;i++) {
        int half=(int)(p->bands[i]/255.0f*mhh); if (half<2) half=2;
        int x0=sx+i*bw, x1=x0+abw;
        int y0=cy-half, y1=cy+half;
        if (x0<0) x0=0; if (x1>DISPLAY_W) x1=DISPLAY_W;
        if (y0<0) y0=0; if (y1>DISPLAY_H) y1=DISPLAY_H;
        for (int y=y0;y<y1;y++)
            for (int x=x0;x<x1;x++)
                buf[y*DISPLAY_W+x]=rgb(255,255,255);
    }
}

/* ════════════════════════════════════════════════════════════════════════════
   Scanline / CRT overlay — subtle horizontal lines for retro-future feel
   ════════════════════════════════════════════════════════════════════════════ */

static void draw_scanlines(uint32_t *buf) {
    for (int y = 1; y < DISPLAY_H; y += 3) {
        for (int x = 0; x < DISPLAY_W; x++) {
            uint32_t p2 = buf[y * DISPLAY_W + x];
            uint8_t r2 = ((p2 >> 16) & 0xFF) * 0.75f;
            uint8_t g2 = ((p2 >>  8) & 0xFF) * 0.75f;
            uint8_t b2 = ((p2      ) & 0xFF) * 0.75f;
            buf[y * DISPLAY_W + x] = rgb(r2, g2, b2);
        }
    }
}

/* Draw a single horizontal rule with glow */
static void draw_rule(uint32_t *buf, int y, uint8_t r, uint8_t g, uint8_t b) {
    for (int x = 0; x < DISPLAY_W; x++) {
        buf[y * DISPLAY_W + x] = rgb(r, g, b);
        if (y > 0)
            buf[(y-1) * DISPLAY_W + x] = rgb(r/3, g/3, b/3);
        if (y < DISPLAY_H-1)
            buf[(y+1) * DISPLAY_W + x] = rgb(r/4, g/4, b/4);
    }
}

/* Draw a vertical rule */
static void draw_vrule(uint32_t *buf, int x, int y0, int y1,
                       uint8_t r, uint8_t g, uint8_t b) {
    for (int y = y0; y < y1; y++) {
        if (y < 0 || y >= DISPLAY_H) continue;
        buf[y * DISPLAY_W + x] = rgb(r, g, b);
        if (x > 0)            buf[y * DISPLAY_W + x-1] = rgb(r/4, g/4, b/4);
        if (x < DISPLAY_W-1)  buf[y * DISPLAY_W + x+1] = rgb(r/4, g/4, b/4);
    }
}

/* Corner brackets [  ] around the album art */
static void draw_corner_brackets(uint32_t *buf, int x0, int y0, int x1, int y1,
                                  int len, uint8_t r, uint8_t g, uint8_t b) {
    /* top-left */
    for (int i=0;i<len;i++) {
        if (y0+0 < DISPLAY_H && x0+i < DISPLAY_W) buf[(y0)*DISPLAY_W + x0+i] = rgb(r,g,b);
        if (y0+1 < DISPLAY_H && x0+i < DISPLAY_W) buf[(y0+1)*DISPLAY_W + x0+i] = rgb(r/2,g/2,b/2);
        if (x0+0 < DISPLAY_W && y0+i < DISPLAY_H) buf[(y0+i)*DISPLAY_W + x0] = rgb(r,g,b);
        if (x0+1 < DISPLAY_W && y0+i < DISPLAY_H) buf[(y0+i)*DISPLAY_W + x0+1] = rgb(r/2,g/2,b/2);
    }
    /* top-right */
    for (int i=0;i<len;i++) {
        if (y0+0 < DISPLAY_H && x1-i >= 0) buf[(y0)*DISPLAY_W + x1-i] = rgb(r,g,b);
        if (y0+1 < DISPLAY_H && x1-i >= 0) buf[(y0+1)*DISPLAY_W + x1-i] = rgb(r/2,g/2,b/2);
        if (x1-0 >= 0 && x1 < DISPLAY_W && y0+i < DISPLAY_H) buf[(y0+i)*DISPLAY_W + x1] = rgb(r,g,b);
        if (x1-1 >= 0 && y0+i < DISPLAY_H) buf[(y0+i)*DISPLAY_W + x1-1] = rgb(r/2,g/2,b/2);
    }
    /* bottom-left */
    for (int i=0;i<len;i++) {
        if (y1 < DISPLAY_H && x0+i < DISPLAY_W) buf[(y1)*DISPLAY_W + x0+i] = rgb(r,g,b);
        if (y1-1 >= 0 && x0+i < DISPLAY_W) buf[(y1-1)*DISPLAY_W + x0+i] = rgb(r/2,g/2,b/2);
        if (x0 < DISPLAY_W && y1-i >= 0 && y1-i < DISPLAY_H) buf[(y1-i)*DISPLAY_W + x0] = rgb(r,g,b);
        if (x0+1 < DISPLAY_W && y1-i >= 0 && y1-i < DISPLAY_H) buf[(y1-i)*DISPLAY_W + x0+1] = rgb(r/2,g/2,b/2);
    }
    /* bottom-right */
    for (int i=0;i<len;i++) {
        if (y1 < DISPLAY_H && x1-i >= 0) buf[(y1)*DISPLAY_W + x1-i] = rgb(r,g,b);
        if (y1-1 >= 0 && x1-i >= 0) buf[(y1-1)*DISPLAY_W + x1-i] = rgb(r/2,g/2,b/2);
        if (x1 < DISPLAY_W && y1-i >= 0) buf[(y1-i)*DISPLAY_W + x1] = rgb(r,g,b);
        if (x1-1 >= 0 && y1-i >= 0) buf[(y1-i)*DISPLAY_W + x1-1] = rgb(r/2,g/2,b/2);
    }
}
/*
 * ════════════════════════════════════════════════════════════════════════════
 * GLITCH CA — v3  (drop-in replacement for the CA block in visualizer_render.c)
 *
 * Architecture
 * ────────────
 *  Each visible entity is a PAIR: one solid + one ghost halo.
 *  g_pairs[i].solid  = solid core
 *  g_pairs[i].ghost  = ghost halo, spatially coupled to its solid parent
 *
 *  The ghost spawns at the SAME seed_col/seed_row as its solid, GHOST_LAG
 *  ticks later, with target_cells = solid.target + small random extra so it
 *  always extends slightly past the solid boundary.  Its frontier picker
 *  uses the parent solid's live[] as a proximity attractor — candidates
 *  that neighbour a solid cell are heavily favoured, so the ghost always
 *  reads as the solid's ragged fringe rather than a separate random blob.
 *
 *  SOLID frontier — center-weighted: inner cells get high oval_weight,
 *                   boundary cells get low weight + small random jitter.
 *                   Fills inward-first → smooth oval body, natural taper.
 *
 *  GHOST frontier — perimeter-hugging: tip-only (no gap-fill), strongly
 *                   biased toward cells adjacent to the solid boundary.
 *                   Allowed into fringe rows for the stringy drip effect.
 *
 *  Ghost presence  — border ~18%, inner ~38%, fill ~55% colour over bg.
 *                    Clearly visible but subordinate to the solid above.
 *
 *  Lifecycle
 *    solid spawns  → ghost_lag = GHOST_LAG; ghost stays DEAD until lag=0
 *    solid DYING   → ghost pushed to DYING on same tick
 *    solid DEAD    → ghost orphaned, finishes dying on its own schedule
 *
 *  Playhead anti-corruption
 *    Cells within KILL_RADIUS px of head_x die every tick.
 *    Spawn rejected if seed col is within SPAWN_DEAD_ZONE of head_col.
 *    Spawn is uniform across the full bar length.
 *
 *  RPi 3B optimisations
 *    · All CA hot paths integer-only (no per-cell float ops in tick)
 *    · Frontier/edge candidate arrays fixed-size on the stack, bounded
 *    · Single xorshift32 RNG — 3 XOR shifts, one register
 *    · 2 pairs × 2 seeds × 384 bytes live[] = ~1.5 kB total state
 * ════════════════════════════════════════════════════════════════════════════
 */

/* ── grid constants ── */
#define CELL_W        6
#define CELL_H        6
#define BAR_FRINGE    10
#define STRIP_TOP     (INFO_PROG_Y - BAR_FRINGE)
#define STRIP_H       (INFO_PROG_H + BAR_FRINGE * 2)
#define GCOLS         (INFO_PROG_W / CELL_W)      /* 580/6 = 96  */
#define GROWS         (STRIP_H    / CELL_H)       /* 26/6  = 4   */
#define GCELLS        (GCOLS * GROWS)             /* 384         */

#define BAR_ROW_MIN   (BAR_FRINGE / CELL_H)                 /* 1 */
#define BAR_ROW_MAX   ((BAR_FRINGE + INFO_PROG_H) / CELL_H) /* 2 */

/* ── tuning ── */
#define MAX_PAIRS        2    /* number of solid+ghost pairs live at once */
#define MIN_LIVE_SOLID   55
#define MAX_LIVE_SOLID   85
#define GHOST_EXTRA      14
#define GHOST_LAG        18   /* ticks after solid spawn before ghost spawns */
#define KILL_RADIUS      6    /* px from head_x → cells die each tick */
#define SPAWN_DEAD_ZONE  2    /* cell columns either side of head_col → no spawn */
#define DRIP_RADIUS      8
#define DRIP_WEIGHT      12

/* ── state machine ── */
typedef enum { SEED_DEAD=0, SEED_GROWING, SEED_ALIVE, SEED_DYING } SeedState;

typedef struct {
    SeedState state;
    uint8_t   live[GCELLS];
    int       live_count;
    int       target_cells;
    int       hold_ticks;
    int       age_ticks;
    int       die_phase;
    int       seed_col;
    int       seed_row;
    uint8_t   cr, cg, cb;
} GlitchSeed;

typedef struct {
    GlitchSeed solid;
    GlitchSeed ghost;
    int        ghost_lag;
    int        drip_col;
    int        drip_dir;   /* -1=left, +1=right, horizontal spread bias */
} GlitchPair;

static GlitchPair g_pairs[MAX_PAIRS];
static float      g_cooldown = 0.0f;

/* ── xorshift32 ── */
static uint32_t g_rng = 0xDEAD1973u;
static inline uint32_t grng(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return g_rng;
}

/* ── cell helpers ── */
static inline int gcell(int col, int row) { return row * GCOLS + col; }
static inline int gcol(int idx)           { return idx % GCOLS; }
static inline int grow_r(int idx)         { return idx / GCOLS; }
static inline int g_is_fringe(int row)    { return (row < BAR_ROW_MIN || row > BAR_ROW_MAX); }

static inline int g_neighbours(const GlitchSeed *s, int col, int row) {
    int n = 0;
    if (col > 0       && s->live[gcell(col-1, row)]) n++;
    if (col < GCOLS-1 && s->live[gcell(col+1, row)]) n++;
    if (row > 0       && s->live[gcell(col, row-1)]) n++;
    if (row < GROWS-1 && s->live[gcell(col, row+1)]) n++;
    return n;
}

static inline int g_nb(const GlitchSeed *s, int nc, int nr) {
    if (nc < 0 || nc >= GCOLS || nr < 0 || nr >= GROWS) return 0;
    return s->live[gcell(nc, nr)];
}

/* ── oval distance weight ──────────────────────────────────────────────────
 *  Stretched ellipse centred on (seed_col, seed_row).
 *  Returns 1-8 graded centre→edge; 0 outside the ellipse.
 *  High weight near centre → frontier fills inward first → smooth oval body.
 */
static inline int g_oval_weight(const GlitchSeed *s, int col, int row) {
    int dc  = col - s->seed_col;
    int dr  = row - s->seed_row;
    int rv2 = 1 + s->target_cells / 12;
    int rh2 = (s->target_cells * s->target_cells * 30) / 100;
    int lhs = dc*dc * rv2 + dr*dr * rh2;
    int rhs = rh2 * rv2;
    if (lhs > rhs * 9 / 4) return 0;
    int dist8 = (lhs * 8) / (rhs + 1);
    return 8 - dist8;
}
/* ── solid frontier picker ─────────────────────────────────────────────────
 *  Dominant term is oval_weight (high near centre → fills inward first).
 *  tip_w allows some gap-fill (nb==2) for a smooth body, but at low weight.
 *  Noise jitter applied only at the outer ring (oval_weight <= 2) so the
 *  interior is deterministically smooth and only the edge looks organic.
 */
#define FRONTIER_MAX 96
static int g_solid_frontier(const GlitchSeed *s) {
    int cands[FRONTIER_MAX];
    int weights[FRONTIER_MAX];
    int nc = 0;

    for (int idx = 0; idx < GCELLS && nc < FRONTIER_MAX; idx++) {
        if (s->live[idx]) continue;
        int c = gcol(idx), r = grow_r(idx);
        if (g_neighbours(s, c, r) == 0) continue;

        int ow = g_oval_weight(s, c, r);
        if (ow == 0) continue;

        /* solid never enters fringe rows */
        if (g_is_fringe(r)) continue;

        int nb = g_neighbours(s, c, r);
        /* gap-fill allowed (nb>=2) to ensure dense uniform body */
        int tip_w = (nb == 1) ? 3 : (nb == 2) ? 2 : (nb >= 3) ? 1 : 0;
        if (tip_w == 0) continue;

        int w = ow * tip_w;
        /* no jitter on interior — keeps body smooth and dense */
        if (w < 1) w = 1;

        cands[nc]   = idx;
        weights[nc] = w;
        nc++;
    }
    if (nc == 0) return -1;

    int total = 0;
    for (int i = 0; i < nc; i++) total += weights[i];
    int pick = (int)(grng() % (uint32_t)total);
    int running = 0;
    for (int i = 0; i < nc; i++) {
        running += weights[i];
        if (pick < running) return cands[i];
    }
    return cands[nc-1];
}

/* ── ghost frontier picker ─────────────────────────────────────────────────
 *  Perimeter-hugging: candidates adjacent to a solid live cell get a large
 *  proximity bonus so the ghost naturally wraps the solid boundary.
 *  tip_w is tip-only (no gap-fill) to keep the ghost stringy.
 *  No fringe penalty — allowed to drip into the fringe rows.
 *
 *  solid_live: parent solid's live[] array (NULL if solid already dead).
 */
static int g_ghost_frontier(const GlitchSeed *s, const uint8_t *solid_live,
                             int drip_col, int drip_dir, int at_target) {
    int cands[FRONTIER_MAX];
    int weights[FRONTIER_MAX];
    int nc = 0;

    for (int idx = 0; idx < GCELLS && nc < FRONTIER_MAX; idx++) {
        if (s->live[idx]) continue;
        int c = gcol(idx), r = grow_r(idx);
        if (g_neighbours(s, c, r) == 0) continue;

        /* clump growth — tips and gap-fill both allowed for natural clusters */
        int nb = g_neighbours(s, c, r);
        int tip_w = (nb == 1) ? 5 : (nb == 2) ? 3 : (nb >= 3) ? 1 : 0;
        if (tip_w == 0) continue;

        /* proximity to solid boundary: each adjacent solid cell adds weight */
        int prox = 0;
        if (solid_live) {
            if (c > 0       && solid_live[gcell(c-1, r)]) prox++;
            if (c < GCOLS-1 && solid_live[gcell(c+1, r)]) prox++;
            if (r > 0       && solid_live[gcell(c, r-1)]) prox++;
            if (r < GROWS-1 && solid_live[gcell(c, r+1)]) prox++;
        }
        /* prox 0 = away from solid (neutral), prox 1-4 = hugging edge */
        int prox_w = prox ? (prox * 4) : 1;

        int drip_w = 1;
        if (g_is_fringe(r)) {
            int dd = c - drip_col;
            if (dd < 0) dd = -dd;
            if (dd <= DRIP_RADIUS)
                drip_w = DRIP_WEIGHT - dd;
        }
        /* once at target size, strongly bias horizontal spread in drip_dir */
        int horiz_w = 1;
        if (at_target && !g_is_fringe(r)) {
            int dc = (drip_dir > 0) ? (c - drip_col) : (drip_col - c);
            if (dc >= 0 && dc < 20)
                horiz_w = 6 - dc / 4;
            if (horiz_w < 1) horiz_w = 1;
        }
        int w = tip_w * prox_w * drip_w * horiz_w;
        w += (int)(grng() % 3) - 1;
        if (w < 1) w = 1;

        cands[nc]   = idx;
        weights[nc] = w;
        nc++;
    }
    if (nc == 0) return -1;

    int total = 0;
    for (int i = 0; i < nc; i++) total += weights[i];
    int pick = (int)(grng() % (uint32_t)total);
    int running = 0;
    for (int i = 0; i < nc; i++) {
        running += weights[i];
        if (pick < running) return cands[i];
    }
    return cands[nc-1];
}

/* ── edge picker (shared) ── */
#define EDGE_MAX 128
static int g_random_edge(const GlitchSeed *s) {
    int cands[EDGE_MAX]; int nc = 0;

    for (int idx = 0; idx < GCELLS; idx++) {
        if (!s->live[idx]) continue;
        int c = gcol(idx), r = grow_r(idx);
        int nb = g_neighbours(s, c, r);
        if (nb >= 3) continue;

        int base_w = (nb == 0) ? 12 : (nb == 1) ? 4 : 1;
        int weight  = g_is_fringe(r) ? base_w * 3 : base_w;
        for (int w = 0; w < weight && nc < EDGE_MAX - 1; w++)
            cands[nc++] = idx;
    }
    if (nc == 0) {
        for (int idx = 0; idx < GCELLS && nc < EDGE_MAX; idx++)
            if (s->live[idx]) cands[nc++] = idx;
    }
    if (nc == 0) return -1;
    return cands[grng() % (uint32_t)nc];
}

/* ── kill cells near playhead ── */
static void g_kill_near_head(GlitchSeed *s, int head_col) {
    for (int idx = 0; idx < GCELLS; idx++) {
        if (!s->live[idx]) continue;
        int d = gcol(idx) - head_col;
        if (d < 0) d = -d;
        if (d * CELL_W <= KILL_RADIUS) {
            s->live[idx] = 0;
            s->live_count--;
        }
    }
    if (s->live_count < 0) s->live_count = 0;
}

/* ── spawn solid (also arms the ghost lag counter) ── */
static void g_spawn_solid(GlitchPair *pair, int head_col,
                           const uint8_t palette[][3]) {
    GlitchSeed *s = &pair->solid;
    memset(s->live, 0, GCELLS);
    s->live_count   = 1;
    s->target_cells = MIN_LIVE_SOLID +
                      (int)(grng() % (MAX_LIVE_SOLID - MIN_LIVE_SOLID + 1));
    s->hold_ticks   = 90 + (int)(grng() % 121);
    s->age_ticks    = 0;
    s->die_phase    = 0;
    s->state        = SEED_GROWING;

    int core = BAR_ROW_MAX - BAR_ROW_MIN + 1;
    if (core < 1) core = 1;
    s->seed_row = BAR_ROW_MIN + (int)(grng() % core);

    int attempts = 0;
    do {
        s->seed_col = 1 + (int)(grng() % (GCOLS - 2));
        int d = s->seed_col - head_col;
        if (d < 0) d = -d;
        if (d > SPAWN_DEAD_ZONE) break;
    } while (++attempts < 32);

    s->live[gcell(s->seed_col, s->seed_row)] = 1;

    int pi = (int)(grng() % NUM_COLORS);
    s->cr = palette[pi][0];
    s->cg = palette[pi][1];
    s->cb = palette[pi][2];
    int total = (int)s->cr + s->cg + s->cb;
    if (total < 180) {
        int boost = (180 - total) / 3;
        s->cr = (uint8_t)(s->cr + boost > 255 ? 255 : s->cr + boost);
        s->cg = (uint8_t)(s->cg + boost > 255 ? 255 : s->cg + boost);
        s->cb = (uint8_t)(s->cb + boost > 255 ? 255 : s->cb + boost);
    }

    pair->ghost_lag   = GHOST_LAG;
    pair->ghost.state = SEED_DEAD;
    int drip_offset = (int)(grng() % 20) - 10;
    pair->drip_col = s->seed_col + drip_offset;
    if (pair->drip_col < 1) pair->drip_col = 1;
    if (pair->drip_col > GCOLS-2) pair->drip_col = GCOLS-2;
    pair->drip_dir = (grng() & 1) ? 1 : -1;
}

/* ── spawn ghost (called internally by g_tick when lag expires) ── */
static void g_spawn_ghost(GlitchPair *pair) {
    const GlitchSeed *solid = &pair->solid;
    GlitchSeed       *g     = &pair->ghost;
    memset(g->live, 0, GCELLS);
    g->live_count   = 1;
    g->target_cells = solid->target_cells + (int)(grng() % (GHOST_EXTRA + 1));
    g->hold_ticks   = solid->hold_ticks;
    g->age_ticks    = 0;
    g->die_phase    = 0;
    g->state        = SEED_GROWING;
    g->seed_col     = solid->seed_col;
    g->seed_row     = solid->seed_row;
    g->live[gcell(g->seed_col, g->seed_row)] = 1;
    g->cr = solid->cr;
    g->cg = solid->cg;
    g->cb = solid->cb;
}

/* ── tick one seed ── */
static void g_tick_seed(GlitchSeed *s, int head_col,
                         int ghost, const uint8_t *solid_live,
                         int drip_col, int drip_dir) {
    if (s->state == SEED_DEAD) return;
    s->age_ticks++;

    g_kill_near_head(s, head_col);
    if (s->live_count <= 0) { s->state = SEED_DEAD; return; }

    if (s->state == SEED_GROWING) {
        if (s->live_count >= s->target_cells) {
            s->state = SEED_ALIVE; s->age_ticks = 0;
        } else {
            int period = ghost ? 3 : 5;
            if ((s->age_ticks % period) == 0) {
                int f = ghost ? g_ghost_frontier(s, solid_live, drip_col, drip_dir,
                                    s->state == SEED_ALIVE)
                              : g_solid_frontier(s);
                if (f < 0) { s->state = SEED_ALIVE; s->age_ticks = 0; }
                else       { s->live[f] = 1; s->live_count++; }
            }
        }
    }
    else if (s->state == SEED_ALIVE) {
        int period = ghost ? 5 : 12;
        if (s->age_ticks % period == 0) {
            int edge = g_random_edge(s);
            if (edge >= 0) {
                s->live[edge] = 0; s->live_count--;
                int f = ghost ? g_ghost_frontier(s, solid_live, drip_col, drip_dir,
                                    s->state == SEED_ALIVE)
                              : g_solid_frontier(s);
                if (f >= 0) { s->live[f] = 1; s->live_count++; }
            }
        }
        if (s->age_ticks >= s->hold_ticks) {
            s->state = SEED_DYING; s->age_ticks = 0; s->die_phase = 0;
        }
    }
    else if (s->state == SEED_DYING) {
        int period = ghost ? 2 : 3;
        s->die_phase = (s->die_phase + 1) % period;
        if (s->die_phase == 0) {
            int edge = g_random_edge(s);
            if (edge >= 0) { s->live[edge] = 0; s->live_count--; }
        }
        if (s->live_count <= 0) s->state = SEED_DEAD;
    }
}

/* ── master tick ── */
static void g_tick(float prog, const uint8_t palette[][3]) {
    int head_col = (int)(prog * GCOLS);

    if (g_cooldown > 0.0f) g_cooldown -= 1.0f;

    if (g_cooldown <= 0.0f && prog > 0.02f && (grng() & 0xFF) < 4) {
        for (int i = 0; i < MAX_PAIRS; i++) {
            if (g_pairs[i].solid.state == SEED_DEAD &&
                g_pairs[i].ghost.state == SEED_DEAD) {
                g_spawn_solid(&g_pairs[i], head_col, palette);
                g_cooldown = 250.0f + (float)(grng() % 201);
                break;
            }
        }
    }

    for (int i = 0; i < MAX_PAIRS; i++) {
        GlitchPair *p = &g_pairs[i];

        /* count down lag; only while solid is still alive */
        if (p->solid.state == SEED_GROWING || p->solid.state == SEED_ALIVE) {
            if (p->ghost.state == SEED_DEAD) {
                if (p->ghost_lag > 0) {
                    p->ghost_lag--;
                } else {
                    g_spawn_ghost(p);
                }
            }
        }

        g_tick_seed(&p->solid, head_col, 0, NULL, p->drip_col, p->drip_dir);

        const uint8_t *sl = (p->solid.state != SEED_DEAD)
                            ? p->solid.live : NULL;
        g_tick_seed(&p->ghost, head_col, 1, sl, p->drip_col, p->drip_dir);

        /* sync AFTER both tick so state is final for this frame */
        if (p->solid.state == SEED_DEAD || p->solid.state == SEED_DYING) {
            if (p->ghost.state == SEED_GROWING || p->ghost.state == SEED_ALIVE) {
                p->ghost.state     = SEED_DYING;
                p->ghost.age_ticks = 0;
                p->ghost.die_phase = 0;
            }
        }
    }
}

/* ── render one seed ── */
static void g_render_seed(const GlitchSeed *s, uint32_t *buf, int ghost) {
    if (s->state == SEED_DEAD) return;

    for (int idx = 0; idx < GCELLS; idx++) {
        if (!s->live[idx]) continue;

        int col = gcol(idx);
        int row = grow_r(idx);
        int px  = INFO_PROG_X + col * CELL_W;
        int py  = STRIP_TOP   + row * CELL_H;

        int face_left  = !g_nb(s, col-1, row);
        int face_right = !g_nb(s, col+1, row);
        int face_up    = !g_nb(s, col,   row-1);
        int face_down  = !g_nb(s, col,   row+1);

        /* solid only: bleed one pixel above/below the strip for a natural edge */
        if (!ghost && face_up) {
            int bpy = py - 1;
            if (bpy >= 0 && bpy < DISPLAY_H) {
                for (int dx = 1; dx < CELL_W-1; dx++) {
                    int fx = px + dx;
                    if ((unsigned)fx < (unsigned)DISPLAY_W)
                        buf[bpy * DISPLAY_W + fx] = rgb(
                            (uint8_t)(s->cr * 0.4f),
                            (uint8_t)(s->cg * 0.4f),
                            (uint8_t)(s->cb * 0.4f));
                }
            }
        }
        if (!ghost && face_down) {
            int bpy = py + CELL_H;
            if (bpy >= 0 && bpy < DISPLAY_H) {
                for (int dx = 1; dx < CELL_W-1; dx++) {
                    int fx = px + dx;
                    if ((unsigned)fx < (unsigned)DISPLAY_W)
                        buf[bpy * DISPLAY_W + fx] = rgb(
                            (uint8_t)(s->cr * 0.4f),
                            (uint8_t)(s->cg * 0.4f),
                            (uint8_t)(s->cb * 0.4f));
                }
            }
        }

        for (int dy = 0; dy < CELL_H; dy++) {
            int fy = py + dy;
            if ((unsigned)fy >= (unsigned)DISPLAY_H) continue;
            for (int dx = 0; dx < CELL_W; dx++) {
                int fx = px + dx;
                if ((unsigned)fx >= (unsigned)DISPLAY_W) continue;

                int is_border = (
                    (dx == 0          && face_left)  ||
                    (dx == CELL_W - 1 && face_right) ||
                    (dy == 0          && face_up)    ||
                    (dy == CELL_H - 1 && face_down)
                );
                int is_inner = !is_border && (
                    (dx == 1          && face_left)  ||
                    (dx == CELL_W - 2 && face_right) ||
                    (dy == 1          && face_up)    ||
                    (dy == CELL_H - 2 && face_down)
                );

                uint32_t dst = buf[fy * DISPLAY_W + fx];
                uint8_t  dr  = (dst >> 16) & 0xFF;
                uint8_t  dg  = (dst >>  8) & 0xFF;
                uint8_t  db  =  dst         & 0xFF;
                uint8_t  out_r, out_g, out_b;

                if (ghost) {
                    /*
                     * Ghost: clearly present but subordinate to solid.
                     * Border ~18% — faint cell outline.
                     * Inner  ~38% — readable tint.
                     * Fill   ~55% — noticeable, still translucent.
                     */
                    if (is_border) {
                        out_r = (uint8_t)(s->cr * 0.18f + dr * 0.72f);
                        out_g = (uint8_t)(s->cg * 0.18f + dg * 0.72f);
                        out_b = (uint8_t)(s->cb * 0.18f + db * 0.72f);
                    } else if (is_inner) {
                        out_r = (uint8_t)(s->cr * 0.38f + dr * 0.62f);
                        out_g = (uint8_t)(s->cg * 0.38f + dg * 0.62f);
                        out_b = (uint8_t)(s->cb * 0.38f + db * 0.62f);
                    } else {
                        out_r = (uint8_t)(s->cr * 0.55f + dr * 0.45f);
                        out_g = (uint8_t)(s->cg * 0.55f + dg * 0.45f);
                        out_b = (uint8_t)(s->cb * 0.55f + db * 0.45f);
                    }
                } else {
                    /*
                     * Solid: EVA-panel — hard dark border, bright fill.
                     */
                    if (is_border) {
                        out_r = s->cr >> 4;
                        out_g = s->cg >> 4;
                        out_b = s->cb >> 4;
                    } else if (is_inner) {
                        out_r = (uint8_t)(s->cr * 0.55f);
                        out_g = (uint8_t)(s->cg * 0.55f);
                        out_b = (uint8_t)(s->cb * 0.55f);
                    } else {
                        out_r = s->cr;
                        out_g = s->cg;
                        out_b = s->cb;
                    }
                }

                buf[fy * DISPLAY_W + fx] = rgb(out_r, out_g, out_b);
            }
        }
    }
}

/* ── master render: ghost layer first, solid on top ── */
static void g_render(uint32_t *buf, int head_x) {
    (void)head_x;
    for (int i = 0; i < MAX_PAIRS; i++) g_render_seed(&g_pairs[i].ghost, buf, 1);
    for (int i = 0; i < MAX_PAIRS; i++) g_render_seed(&g_pairs[i].solid, buf, 0);
}

/* ════════════════════════════════════════════════════════════════════════════
   draw_info_overlay
   ════════════════════════════════════════════════════════════════════════════ */

static void draw_info_overlay(const RenderParams *p, uint32_t *buf) {
    char tmp[128];

    uint8_t ar = p->palette[NUM_COLORS-1][0];
    uint8_t ag = p->palette[NUM_COLORS-1][1];
    uint8_t ab = p->palette[NUM_COLORS-1][2];
    if ((int)ar - (int)ab > 60 || (ar < 60 && ag < 60 && ab < 60)) {
        ar = 0; ag = 220; ab = 200;
    }
    uint8_t dr = ar/3, dg = ag/3, db = ab/3;

    draw_vrule(buf, INFO_TEXT_X - 14,
               INFO_ART_Y - 8, INFO_ART_Y + INFO_ART_SIZE + 8, ar, ag, ab);
    draw_rule(buf, INFO_TITLE_Y - 10, dr, dg, db);
    draw_rule(buf, INFO_ALBUM_Y + 20, dr, dg, db);
    draw_corner_brackets(buf,
        INFO_ART_X - 6,                  INFO_ART_Y - 6,
        INFO_ART_X + INFO_ART_SIZE + 6,  INFO_ART_Y + INFO_ART_SIZE + 6,
        20, ar, ag, ab);

    draw_text(buf, INFO_TEXT_X, INFO_TITLE_Y - 22, "NOW PLAYING", 1, ar, ag, ab, 180);

    truncate_text(p->track_title, tmp, 18);
    draw_text(buf, INFO_TEXT_X, INFO_TITLE_Y, tmp, 3, 255, 255, 255, 255);

    truncate_text(p->track_artist, tmp, 28);
    draw_text(buf, INFO_TEXT_X, INFO_ARTIST_Y, tmp, 2, ar, ag, ab, 230);

    truncate_text(p->track_album, tmp, 40);
    draw_text(buf, INFO_TEXT_X, INFO_ALBUM_Y + 4, tmp, 1, 140, 160, 160, 180);

    /* dark bar track */
    for (int y2 = INFO_PROG_Y; y2 < INFO_PROG_Y + INFO_PROG_H; y2++)
        for (int x2 = INFO_PROG_X; x2 < INFO_PROG_X + INFO_PROG_W; x2++)
            buf[y2*DISPLAY_W+x2] = rgb(20, 30, 35);

    /* progress fill — 30% accent tint leaving a color trace */
    float prog = 0.0f;
    if (p->track_dur_ms > 0)
        prog = (float)p->track_pos_ms / (float)p->track_dur_ms;
    if (prog > 1.0f) prog = 1.0f;
    if (prog < 0.0f) prog = 0.0f;
    int fill_w = (int)(prog * INFO_PROG_W);
    int head_x = INFO_PROG_X + fill_w;

    for (int y2 = INFO_PROG_Y; y2 < INFO_PROG_Y + INFO_PROG_H; y2++) {
        for (int x2 = INFO_PROG_X; x2 < INFO_PROG_X + fill_w; x2++) {
            uint32_t dst = buf[y2*DISPLAY_W+x2];
            buf[y2*DISPLAY_W+x2] = rgb(
                (uint8_t)(ar * 0.30f + ((dst>>16)&0xFF) * 0.70f),
                (uint8_t)(ag * 0.30f + ((dst>> 8)&0xFF) * 0.70f),
                (uint8_t)(ab * 0.30f + ( dst     &0xFF) * 0.70f));
        }
    }

    /* CA tick and render */
    g_tick(prog, p->palette);
    g_render(buf, head_x);

    /* playhead line — always on top */
    if (head_x > INFO_PROG_X && head_x < INFO_PROG_X + INFO_PROG_W) {
        for (int y2 = INFO_PROG_Y - 2; y2 < INFO_PROG_Y + INFO_PROG_H + 2; y2++)
            if (y2 >= 0 && y2 < DISPLAY_H)
                buf[y2*DISPLAY_W+head_x] = rgb(255, 255, 255);
    }

    char elapsed[8], total[8];
    format_time(p->track_pos_ms, elapsed);
    format_time(p->track_dur_ms, total);
    draw_text(buf, INFO_PROG_X, INFO_TIME_Y, elapsed, 1, ar, ag, ab, 220);
    int tlen = 0; while (total[tlen]) tlen++;
    draw_text(buf, INFO_PROG_X + INFO_PROG_W - (tlen*9), INFO_TIME_Y,
              total, 1, dr, dg+40, db+40, 180);

    draw_wave_bars(p, buf);
    overlay_album_art_info(p, buf);
}

/* ════════════════════════════════════════════════════════════════════════════
   Thread worker
   ════════════════════════════════════════════════════════════════════════════ */

static void *render_thread(void *arg) {
    ThreadArgs *ta = (ThreadArgs *)arg;
    while (1) {
        pthread_barrier_wait(&frame_barrier);
        const RenderParams *p = &params_buf[ta->p_idx];
        float dim = (p->display_mode == 1) ? 0.25f : 1.0f;
        if (p->mode == 0)
            render_plasma_chunk(p, ta->buf, ta->start_row, ta->end_row, dim);
        else
            render_stalagmites_chunk(p, ta->buf, ta->start_row, ta->end_row);
        pthread_barrier_wait(&frame_barrier);
        if (ta->thread_id == 0) {
            if (p->display_mode == 1) {
                /* CRT over plasma background first, then UI elements clean on top */
                draw_scanlines(ta->buf);
                draw_info_overlay(p, ta->buf);
            } else {
                /* CRT over plasma first, then freq bars + album art clean on top */
                draw_scanlines(ta->buf);
                if (p->show_bands) draw_freq_bars(p, ta->buf);
                if (p->show_album) overlay_album_art_centered(p, ta->buf);
            }
        }
        pthread_barrier_wait(&frame_barrier);
    }
    return NULL;
}

/* ════════════════════════════════════════════════════════════════════════════
   SHM reader
   ════════════════════════════════════════════════════════════════════════════ */

static void try_update_album(void) {
    uint32_t ready = *(volatile uint32_t *)(shm + OFF_ALBUM_READY);
    if (!ready) { stable_album_ready = 0; return; }
    uint32_t seq = *(volatile uint32_t *)(shm + OFF_SEQ);
    if (seq == stable_album_seq) return;
    int sz = (int)(*(volatile uint32_t *)(shm + OFF_ALBUM_SIZE));
    if (sz < 10 || sz > MAX_ALBUM_SIZE) return;
    uint32_t before = *(volatile uint32_t *)(shm + OFF_ALBUM_READY);
    memcpy(stable_album, (const void *)(shm + OFF_ALBUM_DATA), sz * sz * 3);
    uint32_t after  = *(volatile uint32_t *)(shm + OFF_ALBUM_READY);
    if (before && after) {
        stable_album_ready = 1;
        stable_album_seq   = seq;
        stable_album_size  = sz;
    }
}

static int read_shm(RenderParams *p) {
    uint32_t magic = *(volatile uint32_t *)(shm + OFF_MAGIC);
    if (magic != MAGIC_VAL) return 0;

    memcpy(p->bands, (const void *)(shm + OFF_BANDS), NUM_BANDS * sizeof(float));
    for (int i=0;i<NUM_COLORS;i++) {
        int base = OFF_PALETTE + i*3;
        p->palette[i][0]=shm[base]; p->palette[i][1]=shm[base+1]; p->palette[i][2]=shm[base+2];
    }
    float *abcd=(float *)(shm+OFF_PLASMA_ABCD);
    p->a=abcd[0]; p->b=abcd[1]; p->c=abcd[2]; p->d=abcd[3];
    float *cxcy=(float *)(shm+OFF_PLASMA_CXCY);
    p->cx=cxcy[0]; p->cy=cxcy[1];
    memcpy(p->stala_colors,(const void*)(shm+OFF_STALA_COLORS),7*sizeof(float));
    p->mode       =*(volatile uint32_t*)(shm+OFF_MODE);
    p->show_album =*(volatile uint32_t*)(shm+OFF_SHOW_ALBUM);
    p->show_bands =*(volatile uint32_t*)(shm+OFF_SHOW_BANDS);
    p->album_size =*(volatile uint32_t*)(shm+OFF_ALBUM_SIZE);
    p->bpm        =*(volatile uint32_t*)(shm+OFF_BPM);
    p->display_mode=*(volatile uint32_t*)(shm+OFF_DISPLAY_MODE);

    /* track info */
    memcpy(p->track_title,  (const void*)(shm+OFF_TRACK_TITLE),  63); p->track_title[63]='\0';
    memcpy(p->track_artist, (const void*)(shm+OFF_TRACK_ARTIST), 63); p->track_artist[63]='\0';
    memcpy(p->track_album,  (const void*)(shm+OFF_TRACK_ALBUM_S),63); p->track_album[63]='\0';
    p->track_dur_ms=*(volatile uint32_t*)(shm+OFF_TRACK_DUR_MS);
    p->track_pos_ms=*(volatile uint32_t*)(shm+OFF_TRACK_POS_MS);

    p->album_ready = stable_album_ready;
    if (stable_album_ready && stable_album_size > 0) {
        p->album_size = stable_album_size;
        memcpy(p->album_data, stable_album, stable_album_size * stable_album_size * 3);
    }
    return 1;
}

/* ════════════════════════════════════════════════════════════════════════════
   Framebuffer
   ════════════════════════════════════════════════════════════════════════════ */

static int open_framebuffer(void) {
        fb_fd = open("/dev/fb0", O_RDWR);
    if (fb_fd < 0) {
        #ifdef SDL2_FALLBACK
                printf("/dev/fb0 not available — falling back to SDL2\n");
                SDL_Init(SDL_INIT_VIDEO);
                sdl_window   = SDL_CreateWindow("Hologram Visualizer",
                                SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                DISPLAY_W, DISPLAY_H,
                                SDL_WINDOW_RESIZABLE);
                sdl_renderer = SDL_CreateRenderer(sdl_window, -1,
                                SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
                SDL_RenderSetLogicalSize(sdl_renderer, DISPLAY_W, DISPLAY_H);
                sdl_texture  = SDL_CreateTexture(sdl_renderer,
                                SDL_PIXELFORMAT_ARGB8888,
                                SDL_TEXTUREACCESS_STREAMING,
                                DISPLAY_W, DISPLAY_H);
                return sdl_texture != NULL;
        #else
                perror("open /dev/fb0");
                return 0;
        #endif
    }

    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo)<0) { perror("FBIOGET"); return 0; }
        vinfo.xres=SCREEN_W; vinfo.yres=SCREEN_H;
        vinfo.xres_virtual=SCREEN_W; vinfo.yres_virtual=SCREEN_H*2;
        vinfo.bits_per_pixel=32;
    if (ioctl(fb_fd, FBIOPUT_VSCREENINFO, &vinfo)<0) { perror("FBIOPUT"); return 0; }
        size_t map_size=SCREEN_W*SCREEN_H*2*BYTES_PP;
        fb=(uint32_t*)mmap(NULL,map_size,PROT_READ|PROT_WRITE,MAP_SHARED,fb_fd,0);
    if (fb==MAP_FAILED) { perror("mmap fb"); return 0; }
        memset(fb,0,map_size);
        printf("Framebuffer: %dx%d @ %dbpp\n",vinfo.xres,vinfo.yres,vinfo.bits_per_pixel);
    return 1;
}

static void flip_buffer(uint32_t *rendered_buf) {
    #ifdef SDL2_FALLBACK
    if (sdl_texture) {
        /* Handle window close / escape key */
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) exit(0);
            if (e.type == SDL_KEYDOWN &&
                e.key.keysym.sym == SDLK_ESCAPE) exit(0);
            if (e.type == SDL_KEYDOWN &&
                e.key.keysym.sym == SDLK_F11) {
                Uint32 f = SDL_GetWindowFlags(sdl_window);
                SDL_SetWindowFullscreen(sdl_window,
                    (f & SDL_WINDOW_FULLSCREEN_DESKTOP)
                        ? 0 : SDL_WINDOW_FULLSCREEN_DESKTOP);
            }
        }
        void *pixels; int pitch;
        SDL_LockTexture(sdl_texture, NULL, &pixels, &pitch);
        for (int y = 0; y < DISPLAY_H; y++)
            memcpy((uint8_t*)pixels + y*pitch,
                   rendered_buf + y*DISPLAY_W, DISPLAY_W*BYTES_PP);
        SDL_UnlockTexture(sdl_texture);
        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, sdl_texture, NULL, NULL);
        SDL_RenderPresent(sdl_renderer);
        return;
    }
#endif

    for (int y = 0; y < DISPLAY_H; y++) {
        uint32_t *fb_row = fb + y * SCREEN_W;
        memcpy(fb_row, rendered_buf + y * DISPLAY_W, DISPLAY_W * BYTES_PP);
        uint64_t *fb_tail = (uint64_t *)(fb_row + DISPLAY_W);
        int pad_pixels = SCREEN_W - DISPLAY_W;
        for (int i = 0; i < pad_pixels / 2; i++)
            fb_tail[i] = 0ULL;
        if (pad_pixels & 1)
            fb_row[SCREEN_W - 1] = 0;
    }

}

static int open_shm_file(void) {
    int fd=open(SHM_PATH,O_RDWR);
    if (fd<0) { fprintf(stderr,"Cannot open SHM %s\n",SHM_PATH); return 0; }
    shm=(uint8_t*)mmap(NULL,SHM_SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    close(fd);
    if (shm==MAP_FAILED) { perror("mmap shm"); return 0; }
    printf("SHM mapped: %s (%d bytes)\n",SHM_PATH,SHM_SIZE);
    return 1;
}

/* ════════════════════════════════════════════════════════════════════════════
   Main
   ════════════════════════════════════════════════════════════════════════════ */

int main(void) {
    printf("Hologram Visualizer C Renderer\n");
    printf("Display: %dx%d, %d threads\n",DISPLAY_W,DISPLAY_H,NUM_THREADS);
    build_sin_lut();
    build_dist2_lut();
    if (!open_framebuffer()) return 1;
    printf("Waiting for Python daemon (SHM)...\n");
    while (!open_shm_file()) sleep(1);
    while (*(uint32_t*)(shm+OFF_MAGIC)!=MAGIC_VAL) {
        printf("  waiting for Python to write initial state...\n");
        usleep(500000);
    }
    printf("Python daemon found — starting render loop\n");
    pthread_barrier_init(&frame_barrier, NULL, NUM_THREADS+1);
    read_shm(&params_buf[0]);
    params_buf[0].t=0.0f;
    params_buf[1]=params_buf[0];
    params_idx=0;
    for (int i=0;i<NUM_THREADS;i++) {
        thread_args[i].thread_id=i;
        thread_args[i].start_row=i*ROWS_PER_THREAD;
        thread_args[i].end_row=(i==NUM_THREADS-1)?DISPLAY_H:(i+1)*ROWS_PER_THREAD;
        thread_args[i].p_idx=params_idx;
        thread_args[i].buf=back_buf;
        pthread_create(&threads[i],NULL,render_thread,&thread_args[i]);
        printf("Thread %d: rows %d-%d\n",i,thread_args[i].start_row,thread_args[i].end_row);
    }
    struct timespec frame_start, frame_end;
    const long target_ns = 1000000000L/30;
    while (1) {
        clock_gettime(CLOCK_MONOTONIC,&frame_start);
        int write_idx=params_idx^1;
        try_update_album();
        read_shm(&params_buf[write_idx]);
        params_buf[write_idx].t=params_buf[params_idx].t;
        params_idx=write_idx;
        for (int i=0;i<NUM_THREADS;i++) {
            thread_args[i].p_idx=params_idx;
            thread_args[i].buf=back_buf;
        }
        pthread_barrier_wait(&frame_barrier);
        pthread_barrier_wait(&frame_barrier);
        pthread_barrier_wait(&frame_barrier);
        flip_buffer(back_buf);
        uint32_t *tmp=back_buf; back_buf=front_buf; front_buf=tmp;
        /* in info mode run plasma slower */
        float bpm_factor=params_buf[params_idx].bpm/120.0f;
        float speed=(params_buf[params_idx].display_mode==1)?0.25f:1.0f;
        params_buf[params_idx].t+=bpm_factor*0.07f*speed;
        clock_gettime(CLOCK_MONOTONIC,&frame_end);
        long elapsed=(frame_end.tv_sec-frame_start.tv_sec)*1000000000L
                    +(frame_end.tv_nsec-frame_start.tv_nsec);
        long sleep_ns=target_ns-elapsed;
        if (sleep_ns>0) { struct timespec ts={0,sleep_ns}; clock_nanosleep(CLOCK_MONOTONIC,0,&ts,NULL); }
    }
    return 0;
}