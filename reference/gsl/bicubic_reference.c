/* Reference values of GSL's 2D interpolation as DREAM uses it (ADASRateInterpolator.cpp):
 * bicubic inside the table, bilinear continued outside it (gsl_interp2d_eval_extrap), on a
 * synthetic table with no ADAS data. Writes gsl_bicubic_reference.txt of the set gsl-2.8.
 *
 *   gcc -O0 -I$GSL/include bicubic_reference.c -L$GSL/lib -lgsl -lgslcblas -lm -o bicubic_reference
 *   LD_LIBRARY_PATH=$GSL/lib ./bicubic_reference > gsl_bicubic_reference.txt
 *
 * with GSL=$CONDA_PREFIX, the GSL of the pixi environment of reference/dream
 * (reference/dream/run_all.sh runs these two lines for the set gsl-2.8).
 */
#include <math.h>
#include <stdio.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_version.h>

#define NX 6
#define NY 5

static double table_function(double x, double y) {
    return sin(1.3 * x) * cos(0.7 * y) + 0.3 * x * y * y - 0.1 * x * x;
}

int main(void) {
    const double xa[NX] = {0.0, 0.4, 1.1, 1.5, 2.3, 3.0};
    const double ya[NY] = {-1.0, -0.2, 0.5, 1.6, 2.0};
    double za[NX * NY];
    for (int j = 0; j < NY; j++)
        for (int i = 0; i < NX; i++)
            za[j * NX + i] = table_function(xa[i], ya[j]);
    gsl_interp2d *cubic = gsl_interp2d_alloc(gsl_interp2d_bicubic, NX, NY);
    gsl_interp2d *linear = gsl_interp2d_alloc(gsl_interp2d_bilinear, NX, NY);
    gsl_interp2d_init(cubic, xa, ya, za, NX, NY);
    gsl_interp2d_init(linear, xa, ya, za, NX, NY);
    printf("# GSL %s, bicubic inside and bilinear outside the table as DREAM\n", GSL_VERSION);
    printf("# table: z = sin(1.3x) cos(0.7y) + 0.3 x y^2 - 0.1 x^2\n");
    printf("# x nodes:");
    for (int i = 0; i < NX; i++) printf(" %.17g", xa[i]);
    printf("\n# y nodes:");
    for (int j = 0; j < NY; j++) printf(" %.17g", ya[j]);
    printf("\n# columns: x y z\n");
    const double xs[] = {-0.5, 0.0, 0.2, 0.4, 0.77, 1.1, 1.49, 2.0, 2.3, 2.95, 3.0, 3.6};
    const double ys[] = {-1.7, -1.0, -0.6, -0.2, 0.1, 0.5, 1.2, 1.6, 1.99, 2.0, 2.8};
    for (size_t a = 0; a < sizeof(xs) / sizeof(xs[0]); a++)
        for (size_t b = 0; b < sizeof(ys) / sizeof(ys[0]); b++) {
            double x = xs[a], y = ys[b];
            int outside = x < xa[0] || x > xa[NX - 1] || y < ya[0] || y > ya[NY - 1];
            gsl_interp_accel *xacc = gsl_interp_accel_alloc(), *yacc = gsl_interp_accel_alloc();
            double z = gsl_interp2d_eval_extrap(outside ? linear : cubic, xa, ya, za, x, y, xacc, yacc);
            gsl_interp_accel_free(xacc);
            gsl_interp_accel_free(yacc);
            printf("%.17g %.17g %.17g\n", x, y, z);
        }
    gsl_interp2d_free(cubic);
    gsl_interp2d_free(linear);
    return 0;
}
