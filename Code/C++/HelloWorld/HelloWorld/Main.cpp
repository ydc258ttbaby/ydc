#include<iostream>
#include<string>

int main()
{
	std::string s("abcdefg");
	auto it = s.begin();
	s.erase(it, it+5);
	std::cout<<s<<std::endl;
	std::cin.get();
}