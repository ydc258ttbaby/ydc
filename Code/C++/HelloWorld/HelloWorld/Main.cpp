#include<iostream>
//void PrintName(std::string name) // 可接受左值和右值
//{
//	std::cout<<name<<std::endl;
//}
//void PrintName(std::string& name) // 只接受左值引用，不接受右值
//{
//	std::cout << name << std::endl;
//}
void PrintName(const std::string& name) // 接受左值和右值，把右值当作const lvalue&
{
	std::cout << name << std::endl;
}
//void PrintName(std::string&& name) // 接受右值引用
//{
//	std::cout << name << std::endl;
//}
int main()
{
	std::string firstName = "yang";
	std::string lastName = "dingchao";
	std::string fullName = firstName + lastName;
	PrintName(fullName);
	PrintName(firstName+lastName);
	std::cin.get();
}