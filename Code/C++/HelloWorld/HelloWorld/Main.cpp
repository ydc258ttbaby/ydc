#include<iostream>
//void PrintName(std::string name) // �ɽ�����ֵ����ֵ
//{
//	std::cout<<name<<std::endl;
//}
//void PrintName(std::string& name) // ֻ������ֵ���ã���������ֵ
//{
//	std::cout << name << std::endl;
//}
void PrintName(const std::string& name) // ������ֵ����ֵ������ֵ����const lvalue&
{
	std::cout << name << std::endl;
}
//void PrintName(std::string&& name) // ������ֵ����
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