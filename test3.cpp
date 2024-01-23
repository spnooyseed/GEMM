
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

class Base {
public:
  virtual void f() { cout << "Base::f" << endl; }

  virtual void g() { cout << "Base::g" << endl; }

  virtual void h() { cout << "Base::h" << endl; }
};

typedef void (*Fun)(void);

int main() {
  Base base;

  Fun pFun = NULL;
  std::cout << "&base = " << &base << "\n";
  void *vptrAddr = &base;
  cout << "虚函数表指针地址" << hex << vptrAddr << endl;
  auto vptrTableAddr = *(int *)(vptrAddr);
  cout << "虚函数表地址：" << hex << vptrTableAddr << endl;

  auto func = *((int *)vptrTableAddr);
  pFun = (Fun)(func);
  pFun();

  func = *((int *)vptrTableAddr + sizeof(int) / 2);
  pFun = (Fun)(func);
  pFun();

  func = *((int *)vptrTableAddr + sizeof(int));
  pFun = (Fun)(func);
  pFun();
}
// &base = 0x7fffe4e00da0
// 虚函数表指针地址0x7fffe4e00da0
// 虚函数表地址：400e58
// Base::f
// Base::g
// Base::h