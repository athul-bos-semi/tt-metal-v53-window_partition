#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_common.h"
#include "llk_math_eltwise_binary.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif
