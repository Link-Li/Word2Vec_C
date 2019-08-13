#include <stdio.h>
#include <stdlib.h>

int main()
{
    // printf("Hello world!\n");
    // int a;
    // scanf("%d", &a);
    // printf("%d\n", a);
    // unsigned long long a = 0xffffffffffffffff;
    unsigned long long a = 0x0;

    long long p = a & 0xffff;
    unsigned long long pp = a & 0xffff;

    double b = p / (double)65536;

    a = 0xff;
    p = a & 0xf;

    return 0;
}
