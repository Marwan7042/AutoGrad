#ifndef NN_H
#define NN_H

#include "Tensor.h"
#include <random>
#include <cmath>

namespace mstd {
    namespace nn {
        template <typename T>
        class Dense {
        public:
            Tensor<T> weight;
            Tensor<T> bias;

            // Initialize the Dense layer with random weights
            Dense(size_t inputs, size_t neurons) {
                mstd::vector<size_t> w_shape(2);
                w_shape[0] = inputs; 
                w_shape[1] = neurons;
                weight = Tensor<T>(w_shape);

                mstd::vector<size_t> b_shape(2);
                b_shape[0] = 1; 
                b_shape[1] = neurons;
                bias = Tensor<T>(b_shape);
                
                weight.ctx->requires_grad = true;
                bias.ctx->requires_grad = true;

                // Mathematical Xavier Initialization for fast learning
                std::random_device rd;
                std::mt19937 gen(rd());
                float limit = std::sqrt(6.0f / (inputs + neurons));
                std::uniform_real_distribution<float> dis(-limit, limit);

                for (size_t i = 0; i < weight.data->size(); i++) {
                    (*weight.data)[i] = dis(gen);
                }
                for (size_t i = 0; i < bias.data->size(); i++) {
                    (*bias.data)[i] = 0.0f; // biases start at 0
                }
            }

            // The Forward Pass for this layer!
            Tensor<T> operator()(const Tensor<T>& x) {
                // (1, in) * (in, out) = (1, out)
                return (x * weight) + bias;
            }
            
            // Helper to get all parameters so our Optimizer can update them
            mstd::vector<Tensor<T>*> parameters() {
                mstd::vector<Tensor<T>*> params(2);
                params[0] = &weight;
                params[1] = &bias;
                return params;
            }
        };

        template <typename T>
        class MSELossNode : public AutogradNode<T> {
        private:
            Tensor<T> pred, target, out;
        public:
            MSELossNode(Tensor<T> p, Tensor<T> t, Tensor<T> o) : pred(p), target(t), out(o) {}
            
            mstd::vector<Tensor<T>> get_parents() override {
                mstd::vector<Tensor<T>> parents(2);
                parents[0] = pred; parents[1] = target;
                return parents;
            }

            void backward() override {
                if (pred.ctx->requires_grad) {
                    if (!pred.ctx->grad) pred.ctx->grad = std::make_shared<Tensor<T>>(pred._shape);
                    for (size_t i = 0; i < pred.ctx->grad->data->size(); i++) {
                        // The derivative of (pred - target)^2 is 2 * (pred - target)
                        (*pred.ctx->grad->data)[i] += 2.0f * ((*pred.data)[i] - (*target.data)[i]) * (*out.ctx->grad->data)[0];
                    }
                }
            }
        };

        template <typename T>
        class MSELoss {
        public:
            Tensor<T> operator()(const Tensor<T>& predictions, const Tensor<T>& targets) {
                mstd::vector<size_t> out_shape(2); out_shape[0] = 1; out_shape[1] = 1;
                Tensor<T> result(out_shape);
                
                float sum = 0.0f;
                for (size_t i = 0; i < predictions.data->size(); i++) {
                    float err = (*predictions.data)[i] - (*targets.data)[i];
                    sum += err * err;
                }
                (*result.data)[0] = sum / predictions.data->size(); // Mean Squared Error

                result.ctx->requires_grad = predictions.ctx->requires_grad;
                if (result.ctx->requires_grad) {
                    result.ctx->creator = std::make_shared<MSELossNode<T>>(predictions, targets, result);
                }
                return result;
            }
        };

        template <typename T>
        class CrossEntropyLossNode : public AutogradNode<T> {
        private:
            Tensor<T> logits, target, out;
            mstd::vector<T> softmax_probs;
        public:
            CrossEntropyLossNode(Tensor<T> l, Tensor<T> t, Tensor<T> o, mstd::vector<T> probs) 
                : logits(l), target(t), out(o), softmax_probs(probs) {}
            
            mstd::vector<Tensor<T>> get_parents() override {
                mstd::vector<Tensor<T>> parents(2);
                parents[0] = logits; parents[1] = target;
                return parents;
            }

            void backward() override {
                if (logits.ctx->requires_grad) {
                    if (!logits.ctx->grad) logits.ctx->grad = std::make_shared<Tensor<T>>(logits._shape);
                    
                    // The derivative of CrossEntropy with Softmax is elegantly just: (Prob - Target)
                    for (size_t i = 0; i < logits.ctx->grad->data->size(); i++) {
                        (*logits.ctx->grad->data)[i] += (softmax_probs[i] - (*target.data)[i]) * (*out.ctx->grad->data)[0];
                    }
                }
            }
        };

        template <typename T>
        class CrossEntropyLoss {
        public:
            Tensor<T> operator()(const Tensor<T>& logits, const Tensor<T>& targets) {
                mstd::vector<size_t> out_shape(2); out_shape[0] = 1; out_shape[1] = 1;
                Tensor<T> result(out_shape);
                
                // 1. Calculate Softmax
                mstd::vector<T> probs(logits.data->size());
                T max_logit = (*logits.data)[0];
                for (size_t i = 1; i < logits.data->size(); i++) {
                    if ((*logits.data)[i] > max_logit) max_logit = (*logits.data)[i];
                }
                
                T sum_exp = 0.0f;
                for (size_t i = 0; i < logits.data->size(); i++) {
                    probs[i] = std::exp((*logits.data)[i] - max_logit); // subtract max for numerical stability
                    sum_exp += probs[i];
                }
                
                T loss = 0.0f;
                for (size_t i = 0; i < logits.data->size(); i++) {
                    probs[i] /= sum_exp;
                    
                    // 2. Calculate Cross Entropy Loss: -sum(target * log(prob))
                    if ((*targets.data)[i] > 0.5f) { // If this is the target class (1.0)
                        loss -= std::log(probs[i] + 1e-7f); // add small epsilon to prevent log(0)
                    }
                }
                
                (*result.data)[0] = loss;

                result.ctx->requires_grad = logits.ctx->requires_grad;
                if (result.ctx->requires_grad) {
                    result.ctx->creator = std::make_shared<CrossEntropyLossNode<T>>(logits, targets, result, probs);
                }
                return result;
            }
        };
    }
}
#endif
