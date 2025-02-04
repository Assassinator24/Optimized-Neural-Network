#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    int input_size, output_size;
    float *weights;
    float *biases;
    float *activation;
    float *input;
    float *weight_grads;
    float *bias_grads;
} DenseLayer;

DenseLayer create_layer(int in_size, int out_size) {
    DenseLayer layer;
    layer.input_size = in_size;
    layer.output_size = out_size;
    
    layer.weights = malloc(in_size * out_size * sizeof(float));
    layer.biases = malloc(out_size * sizeof(float));
    layer.activation = malloc(out_size * sizeof(float));
    layer.input = malloc(in_size * sizeof(float));
    layer.weight_grads = malloc(in_size * out_size * sizeof(float));
    layer.bias_grads = malloc(out_size * sizeof(float));

    // He initialization
    float std = sqrtf(2.0f / in_size);
    for (int i = 0; i < in_size * out_size; i++) {
        layer.weights[i] = (float)rand() / RAND_MAX * 2 * std - std;
    }
    for (int i = 0; i < out_size; i++) {
        layer.biases[i] = 0.0f;
    }
    
    return layer;
}

void forward(DenseLayer *layer, float *input) {
    // Store input for backprop
    for (int i = 0; i < layer->input_size; i++) {
        layer->input[i] = input[i];
    }

    // Compute activation
    for (int i = 0; i < layer->output_size; i++) {
        layer->activation[i] = layer->biases[i];
        for (int j = 0; j < layer->input_size; j++) {
            layer->activation[i] += input[j] * layer->weights[i * layer->input_size + j];
        }
        // ReLU
        layer->activation[i] = fmaxf(0.0f, layer->activation[i]);
    }
}

void backward(DenseLayer *layer, float *upstream_grad, float lr) {
    // Compute ReLU gradient
    float *relu_grad = malloc(layer->output_size * sizeof(float));
    for (int i = 0; i < layer->output_size; i++) {
        relu_grad[i] = (layer->activation[i] > 0) ? 1.0f : 0.0f;
        relu_grad[i] *= upstream_grad[i];
    }

    // Calculate weight gradients
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            int idx = i * layer->input_size + j;
            layer->weight_grads[idx] = layer->input[j] * relu_grad[i];
        }
    }

    // Calculate bias gradients
    for (int i = 0; i < layer->output_size; i++) {
        layer->bias_grads[i] = relu_grad[i];
    }

    // Calculate downstream gradient
    float *downstream_grad = malloc(layer->input_size * sizeof(float));
    for (int j = 0; j < layer->input_size; j++) {
        downstream_grad[j] = 0.0f;
        for (int i = 0; i < layer->output_size; i++) {
            downstream_grad[j] += layer->weights[i * layer->input_size + j] * relu_grad[i];
        }
    }

    // Update weights
    for (int i = 0; i < layer->output_size * layer->input_size; i++) {
        layer->weights[i] -= lr * layer->weight_grads[i];
    }

    // Update biases
    for (int i = 0; i < layer->output_size; i++) {
        layer->biases[i] -= lr * layer->bias_grads[i];
    }

    free(relu_grad);
    free(downstream_grad);
}

float mse_loss(float *output, float *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = output[i] - target[i];
        loss += diff * diff;
    }
    return loss / size;
}

void calculate_output_grad(float *output, float *target, float *grad, int size) {
    for (int i = 0; i < size; i++) {
        grad[i] = 2.0f * (output[i] - target[i]) / size;
    }
}

int main() {
    srand(time(NULL));
    
    // Network architecture
    DenseLayer hidden = create_layer(2, 4);
    DenseLayer output = create_layer(4, 1);
    
    // Training parameters
    float learning_rate = 0.01f;
    int epochs = 1000;
    
    // Sample data
    float input[2] = {0.5f, -0.3f};
    float target[1] = {1.0f};
    
    // Training loop
    for (int i = 0; i < epochs; i++) {
        // Forward pass
        forward(&hidden, input);
        forward(&output, hidden.activation);
        
        // Calculate loss
        float loss = mse_loss(output.activation, target, 1);
        
        // Backward pass
        float output_grad[1];
        calculate_output_grad(output.activation, target, output_grad, 1);
        backward(&output, output_grad, learning_rate);
        backward(&hidden, output.weight_grads, learning_rate);
        
        if (i % 100 == 0) {
            printf("Epoch %d, Loss: %.4f\n", i, loss);
        }
    }
    
    // Cleanup
    free(hidden.weights); free(hidden.biases); free(hidden.activation); 
    free(hidden.input); free(hidden.weight_grads); free(hidden.bias_grads);
    free(output.weights); free(output.biases); free(output.activation);
    free(output.input); free(output.weight_grads); free(output.bias_grads);
    
    return 0;
}
