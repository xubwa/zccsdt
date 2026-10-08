! Element type of a compilation of the type-generic sources (*.inc):
!   XZ_REAL defined   real(8) elements,    names ..._d, 1 word per element
!   otherwise         complex(8) elements, names ..._z, 2 words per element
! N(name) is the name of a routine (or of a C kernel) for the element type.
#define PASTE(a) a
#undef ZT
#undef NW
#undef N
#undef CONJ
#undef XGEMM
#ifdef XZ_REAL
#define ZT real(8)
#define NW 1
#define N(x) PASTE(x)_d
#define CONJ(x) (x)
#else
#define ZT complex(8)
#define NW 2
#define N(x) PASTE(x)_z
#define CONJ(x) conjg(x)
#endif
