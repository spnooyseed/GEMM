#include <iostream>
template <typename T>
T add(T a, T b) {
    return a + b;
}

int main() {
        int result1 = add(5, 10);       // 在调用点进行类型实例化，可以内联
        double result2 = add(3.5, 7.2); // 在调用点进行类型实例化，可以内联
        std::cout << result1 << " " << result2 << "\n" ;
    return 0;
}