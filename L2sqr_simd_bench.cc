#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <x86intrin.h>

using namespace std;


// https://github.com/v1k1nghawk/mariadb_server/blob/bb-11.4-vec-preview/sql/item_vectorfunc.cc#L55
double euclidean_vec_distance(float *v1, float *v2, size_t v_len)
{
  float *p1= v1;
  float *p2= v2;
  double d= 0;
  for (size_t i= 0; i < v_len; p1++, p2++, i++)
  {
    float dist= *p1 - *p2;
    d+= dist * dist;
  }
  return d;
}

double euclidean_vec_distance_precise(float *v1, float *v2, size_t v_len)
{
  float *p1= v1;
  float *p2= v2;
  double d= 0;
  for (size_t i= 0; i < v_len; p1++, p2++, i++)
  {
    float dist= *p1 - *p2;
    d+= static_cast<double>(dist) * dist;
  }
  return d;
}

double euclidean_vec_distance_sse(float *v1, float *v2, size_t v_len)
{
    float *p1 = v1;
    float *p2 = v2;
    __m128d d = _mm_setzero_pd();

    size_t i;
    // process 4 elems per loop
    for(i = 0; i < v_len - 3; i += 4)
    {
        const __m128 a = _mm_loadu_ps(p1 + i);
        const __m128 b = _mm_loadu_ps(p2 + i);

        const __m128d c = _mm_cvtps_pd(_mm_sub_ps(a, b)); // c = a - b
        const __m128d dist = _mm_mul_pd(c, c); // dist = c * c

        d = _mm_add_pd(d, dist); // d += dist
    }

    // process vectors' tail (3 elems at max)
    for(; i < v_len; ++i)
    {
        const __m128 a = _mm_load_ss(p1 + i);
        const __m128 b = _mm_load_ss(p2 + i);

        const __m128d c = _mm_cvtps_pd(_mm_sub_ps(a, b));
        const __m128d dist = _mm_mul_pd(c, c);

        d = _mm_add_pd(d, dist);
    }

    d = _mm_hadd_pd(d, d);
    d = _mm_hadd_pd(d, d);
    return _mm_cvtsd_f64(_mm_unpackhi_pd(d, d));
}

double run_experiment(uint r, double (*f)(float*, float*, size_t), float *v1, float *v2, size_t v_len) {
    double ret_vals[r];
    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (uint i = 0; i < r; ++i) {
        float v1_mod[v_len];
        for (uint i = 0; i < v_len; ++i) {
            v1_mod[i] = v1[i] + 1.;
        }
        ret_vals[i] = f(v1_mod, v2, v_len);
    }
    gettimeofday(&end, NULL);
    printf("ret_vals[0]: %f\n", ret_vals[0]);
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1000000.0;
}


int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        printf("usage: %s [vec_size] [rounds], \n", argv[0]);
        return -1;
    }
    const uint vec_size = atoi(argv[1]);
    const uint rounds = atoi(argv[2]);

    float vec1[vec_size];
    float vec2[vec_size];
    for (uint i = 0; i < vec_size; ++i)
    {
        vec1[i] = static_cast<float>(i);
        vec2[i] = static_cast<float>(vec_size-i);
    }

    printf("orig        : elapsed time (sec): %f\n", run_experiment(rounds, euclidean_vec_distance, vec1, vec2, vec_size));
    printf("orig precise: elapsed time (sec): %f\n", run_experiment(rounds, euclidean_vec_distance_precise, vec1, vec2, vec_size));
    printf("sse         : elapsed time (sec): %f\n", run_experiment(rounds, euclidean_vec_distance_sse, vec1, vec2, vec_size));

    return 0;
}
