#ifndef Tensor_H
#define Tensor_H

#include "vector.h"
#include "unordered_map.h"
#include <stdexcept>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace mstd{
    template <typename T>                                                                                                   
    class AutogradNode; 

    template <typename T>
    struct GPUData {
        T* ptr = nullptr;
        size_t size = 0;
        GPUData(size_t s) : size(s) {
            cudaMalloc((void**)&ptr, size * sizeof(T));
        }
        ~GPUData() {
            if (ptr) cudaFree(ptr);
        }
    };

    template <typename T>
    class Tensor;

    template <typename T>
    struct AutogradContext {
        bool requires_grad = false;
        std::shared_ptr<Tensor<T>> grad = nullptr;
        std::shared_ptr<AutogradNode<T>> creator = nullptr;
    };

    template <typename T>
    class Tensor {
    public:
        mstd::vector<size_t> _shape;
        mstd::vector<size_t> _strides;
        std::shared_ptr<mstd::vector<T>> data;
        std::shared_ptr<AutogradContext<T>> ctx;
        
        bool is_cuda = false;
        std::shared_ptr<GPUData<T>> gpu_data = nullptr;

        Tensor() { ctx = std::make_shared<AutogradContext<T>>(); }
        Tensor(mstd::vector<size_t> s, mstd::vector<size_t> st, std::shared_ptr<mstd::vector<T>> d, std::shared_ptr<AutogradContext<T>> c, bool cuda=false, std::shared_ptr<GPUData<T>> gd=nullptr) 
            : _shape(s), _strides(st), data(d), ctx(c), is_cuda(cuda), gpu_data(gd) {}
    public:
        Tensor(mstd::vector<size_t> shape);

        Tensor<T> to(const std::string& device) const;

        T& operator()(const mstd::vector<size_t>& dims);
        
        Tensor<T> transpose() const;    
        Tensor<T> operator+(const Tensor<T>& other) const;
        Tensor<T> operator-(const Tensor<T>& other) const;
        Tensor<T> operator*(const Tensor<T>& other) const;
        Tensor<T> relu() const;

        void backward();

        void print() const;
    };

    template <typename T>
    class AutogradNode {
    public:
        virtual void backward() = 0;
        virtual mstd::vector<Tensor<T>> get_parents() = 0;
        virtual ~AutogradNode() = default;
    };
    
    template <typename T>
    class AddNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b, out;
    public:
        AddNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out(out) {}

        mstd::vector<Tensor<T>> get_parents() override {
            mstd::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) a.ctx->grad = std::make_shared<Tensor<T>>(a._shape);
                for (size_t i = 0; i < a.ctx->grad->data->size(); i++) (*a.ctx->grad->data)[i] += (*out.ctx->grad->data)[i];
            }
            if (b.ctx->requires_grad) {
                if (!b.ctx->grad) b.ctx->grad = std::make_shared<Tensor<T>>(b._shape);
                for (size_t i = 0; i < b.ctx->grad->data->size(); i++) (*b.ctx->grad->data)[i] += (*out.ctx->grad->data)[i];
            }
        }
    };

    template <typename T>
    class SubNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b, out;
    public:
        SubNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out(out) {}

        mstd::vector<Tensor<T>> get_parents() override {
            mstd::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) a.ctx->grad = std::make_shared<Tensor<T>>(a._shape);
                for (size_t i = 0; i < a.ctx->grad->data->size(); i++) (*a.ctx->grad->data)[i] -= (*out.ctx->grad->data)[i];
            }
            if (b.ctx->requires_grad) {
                if (!b.ctx->grad) b.ctx->grad = std::make_shared<Tensor<T>>(b._shape);
                for (size_t i = 0; i < b.ctx->grad->data->size(); i++) (*b.ctx->grad->data)[i] -= (*out.ctx->grad->data)[i];
            }
        }
    };

    template <typename T>
    class MulNode : public AutogradNode<T> {
    private:
        Tensor<T> a, b, out;
    public:
        MulNode(Tensor<T> a, Tensor<T> b, Tensor<T> out) : a(a), b(b), out(out) {}

        mstd::vector<Tensor<T>> get_parents() override {
            mstd::vector<Tensor<T>> parents(2);
            parents[0] = a;
            parents[1] = b;
            return parents;
        }

        void backward() override {
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) a.ctx->grad = std::make_shared<Tensor<T>>(a._shape);
                Tensor<T> a_grad_update = *(out.ctx->grad) * b.transpose();
                *(a.ctx->grad) = *(a.ctx->grad) + a_grad_update;
            }
            if (b.ctx->requires_grad) {
                if (!b.ctx->grad) b.ctx->grad = std::make_shared<Tensor<T>>(b._shape);
                Tensor<T> b_grad_update = a.transpose() * *(out.ctx->grad);
                *(b.ctx->grad) = *(b.ctx->grad) + b_grad_update;
            }
        }
    };

    template <typename T>
    class ReLU : public AutogradNode<T> {
    private:
        Tensor<T> a, out;
    public:
        ReLU(Tensor<T> a, Tensor<T> out) : a(a), out(out) {}

        mstd::vector<Tensor<T>> get_parents() override {
            mstd::vector<Tensor<T>> parents(1);
            parents[0] = a;
            return parents;
        }

        void backward() override {
            if (a.ctx->requires_grad) {
                if (!a.ctx->grad) a.ctx->grad = std::make_shared<Tensor<T>>(a._shape);
                for (size_t i = 0; i < a.ctx->grad->data->size(); i++) if ((*a.data)[i] > 0) (*a.ctx->grad->data)[i] += (*out.ctx->grad->data)[i];
            }
        }
    };

}

#endif