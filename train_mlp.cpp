#include "nn.h"
#include "optim.h"
#include <iostream>

using namespace mstd;

// Your very first Multi-Layer Perceptron!
class MLP {
public:
    nn::Dense<float> layer1;
    nn::Dense<float> layer2;
    nn::Dense<float> layer3;

    MLP() : 
        layer1(2, 8),  // 2 inputs, 8 hidden neurons
        layer2(8, 8),  // 8 hidden, 8 hidden neurons
        layer3(8, 1)   // 8 hidden, 1 output neuron
    {}

    // The Magic Forward Pass using your ReLU!
    Tensor<float> operator()(const Tensor<float>& x) {
        Tensor<float> h1 = layer1(x).relu(); // Hidden Layer 1 + Activation
        Tensor<float> h2 = layer2(h1).relu(); // Hidden Layer 2 + Activation
        return layer3(h2);                    // Output Layer (No activation, just raw prediction)
    }

    vector<Tensor<float>*> parameters() {
        vector<Tensor<float>*> params(6);
        auto p1 = layer1.parameters();
        auto p2 = layer2.parameters();
        auto p3 = layer3.parameters();
        params[0] = p1[0]; params[1] = p1[1];
        params[2] = p2[0]; params[3] = p2[1];
        params[4] = p3[0]; params[5] = p3[1];
        return params;
    }
};

int main() {
    std::cout << "Training MLP on the XOR Problem (Impossible for a single layer!)..." << std::endl;
    
    MLP model;
    
    // XOR Dataset
    float X_data[4][2] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}};
    float Y_data[4]    = {0.0f,         1.0f,         1.0f,         0.0f};
    
    vector<size_t> x_shape(2); x_shape[0] = 1; x_shape[1] = 2;
    vector<size_t> y_shape(2); y_shape[0] = 1; y_shape[1] = 1;
    
    Tensor<float> X(x_shape);
    Tensor<float> Y_true(y_shape);
    
    vector<size_t> idx00(2); idx00[0] = 0; idx00[1] = 0;
    vector<size_t> idx01(2); idx01[0] = 0; idx01[1] = 1;

    float learning_rate = 0.005f; // Faster learning rate
    
    // Abstracted PyTorch-style architecture!
    optim::SGD<float> optimizer(model.parameters(), learning_rate);
    nn::MSELoss<float> criterion;

    for (int epoch = 0; epoch <= 3000; epoch++) {
        float epoch_loss = 0.0f;
        
        for (int i = 0; i < 4; i++) {
            // Load Batch
            X(idx00) = X_data[i][0];
            X(idx01) = X_data[i][1];
            Y_true(idx00) = Y_data[i];
            
            // 1. Forward Pass
            Tensor<float> Y_pred = model(X);
            
            // 2. Calculate Loss
            Tensor<float> Loss = criterion(Y_pred, Y_true);
            epoch_loss += Loss(idx00);
            
            // 3. Clear Gradients
            optimizer.zero_grad();
            
            // 4. Backward Pass (Calculate Gradients)
            Loss.backward();
            
            // 5. Optimizer Step (Update Weights)
            optimizer.step();
        }
        
        if (epoch % 500 == 0) {
            std::cout << "Epoch " << epoch << " | Loss: " << epoch_loss / 4.0f << std::endl;
        }
    }
    
    std::cout << "\n--- Final Predictions ---" << std::endl;
    for (int i = 0; i < 4; i++) {
        X(idx00) = X_data[i][0];
        X(idx01) = X_data[i][1];
        Tensor<float> Y_pred = model(X);
        std::cout << "Input [" << X_data[i][0] << ", " << X_data[i][1] << "] => Output: " << Y_pred(idx00) << std::endl;
    }
    
    return 0;
}
