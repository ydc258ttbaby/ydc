// TestOfMutable.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

struct A
{
    int a1;
    mutable int a2;
};
class B
{
public:
    int b1;
    mutable int b2;
    int modifyVariant(int i) const
    {
        b1 = 1;
        b2 = 2;
    }
};
int main()
{
    const A AInstance = {1,2};
    AInstance.a1 = 1;
    AInstance.a2 = 2;
}

