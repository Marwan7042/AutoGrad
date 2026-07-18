#include "vecore/Tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>

// Helper to fill a 2D tensor easily
void fill_tensor(vc::Tensor<float>& t, float val, size_t rows, size_t cols) {
    vc::vector<size_t> d(2);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            d[0] = i; d[1] = j;
            t(d) = val;
        }
    }
}

// Helper to check a 2D tensor's values
void assert_tensor_val(vc::Tensor<float>& t, float expected, size_t rows, size_t cols) {
    vc::vector<size_t> d(2);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            d[0] = i; d[1] = j;
            float val = t(d);
            if (std::abs(val - expected) > 1e-4) {
                std::cout << "Assertion Failed! Expected " << expected << " but got " << val << std::endl;
                assert(false);
            }
        }
    }
}

void test_forward_add() {
    vc::vector<size_t> shape(2);
    shape[0] = 2; shape[1] = 2;
    
    vc::Tensor<float> A(shape);
    vc::Tensor<float> B(shape);
    
    fill_tensor(A, 2.0f, 2, 2);
    fill_tensor(B, 3.0f, 2, 2);
    
    vc::Tensor<float> C = A + B;
    assert_tensor_val(C, 5.0f, 2, 2);
    std::cout << "[PASS] Forward Addition" << std::endl;
}

void test_forward_matmul() {
    vc::vector<size_t> shape(2);
    shape[0] = 2; shape[1] = 2;
    
    vc::Tensor<float> A(shape);
    vc::Tensor<float> B(shape);
    
    fill_tensor(A, 2.0f, 2, 2); // [[2, 2], [2, 2]]
    fill_tensor(B, 3.0f, 2, 2); // [[3, 3], [3, 3]]
    
    vc::Tensor<float> C = A * B;
    // 2*3 + 2*3 = 12
    assert_tensor_val(C, 12.0f, 2, 2);
    std::cout << "[PASS] Forward Matrix Multiplication (OpenBLAS)" << std::endl;
}

void test_autograd_add() {
    vc::vector<size_t> shape(2);
    shape[0] = 2; shape[1] = 2;
    
    vc::Tensor<float> A(shape);
    vc::Tensor<float> B(shape);
    A.ctx->requires_grad = true; // Track gradients!
    
    fill_tensor(A, 2.0f, 2, 2);
    fill_tensor(B, 3.0f, 2, 2);
    
    vc::Tensor<float> C = A + B;
    C.backward();
    
    // Derivative of addition is 1
    assert_tensor_val(*A.ctx->grad, 1.0f, 2, 2);
    std::cout << "[PASS] Autograd Addition" << std::endl;
}

void test_autograd_complex() {
    vc::vector<size_t> shape(2);
    shape[0] = 2; shape[1] = 2;
    
    vc::Tensor<float> A(shape);
    vc::Tensor<float> B(shape);
    vc::Tensor<float> C(shape);
    
    A.ctx->requires_grad = true;
    B.ctx->requires_grad = true;
    
    fill_tensor(A, 2.0f, 2, 2);
    fill_tensor(B, 3.0f, 2, 2);
    fill_tensor(C, 4.0f, 2, 2);
    
    // D = (A * B) + C
    vc::Tensor<float> M = A * B;
    vc::Tensor<float> D = M + C;
    
    D.backward();
    
    // D = A*B + C. 
    // dD/dA = B. Since D is a 2x2 of 1s (gradient seed), A.grad = D.grad * B.T() = 1 * B.T()
    // Since B is all 3s, B.T() is all 3s. So 1 * 3 + 1 * 3 = 6
    assert_tensor_val(*A.ctx->grad, 6.0f, 2, 2);
    assert_tensor_val(*B.ctx->grad, 4.0f, 2, 2); // A.T() * D.grad = 2*1 + 2*1 = 4
    
    std::cout << "[PASS] Autograd Complex (MatMul + Add Graph)" << std::endl;
}

int main() {
    std::cout << "Running DeepEngine Test Suite..." << std::endl;
    std::cout << "================================" << std::endl;
    
    test_forward_add();
    test_forward_matmul();
    test_autograd_add();
    test_autograd_complex();
    
    std::cout << "================================" << std::endl;
    std::cout << "ALL TESTS PASSED SUCCESSFULLY! BULLETPROOF!" << std::endl;
    
    return 0;
}
