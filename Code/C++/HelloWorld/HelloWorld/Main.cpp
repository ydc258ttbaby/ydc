#include<iostream>
class A
{
public:
	void print()
	{
		std::cout << "A\n";
	}
	virtual void Vprint()
	{
		std::cout << "VA\n";
	}
};
class B :public A
{
public:
	void print()
	{
		std::cout << "B\n";
	}
	void Vprint() override
	{
		std::cout << "VB\n";
	}
};
int main()
{
	A* ab = new B();
	B* bb = new B();
	ab->print();
	bb->print();
	ab->Vprint();
	bb->Vprint();
	std::cin.get();






}

}

}