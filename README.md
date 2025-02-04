**Lightweight Neural Network in C**  

**Description**  
A simple neural network implementation from scratch in C, featuring:  
- Linear time complexity (O(n) operations relative to parameters)  
- ReLU activation and MSE loss  
- SGD optimization with backpropagation  
- Memory-efficient implementation  

**Features**  
- Layer-wise forward/backward passes  
- He weight initialization  
- Automatic gradient computation  
- Minimal dependencies (standard C library only)  

**Build & Run**  
1. **Requirements**: GCC compiler, math library (-lm)  
2. **Compile**:  
   ```  
   gcc -O3 -Wall -Wextra -o neural_net neural_net.c -lm  
   ```  
3. **Execute**:  
   ```  
   ./neural_net  
   ```  

**Implementation Details**  
- **Layer Structure**: Stores weights, biases, activations, and gradients in contiguous arrays  
- **Forward Pass**:  
  - Matrix-vector operations with ReLU activation  
  - O(n) time complexity per layer  
- **Backward Pass**:  
  - Chain rule implementation for gradient calculations  
  - Weight/bias updates via SGD  
- **Memory**: Pre-allocated buffers for intermediate values  

**Example Usage**  
The included sample trains a 2-layer network on synthetic data:  
- Input: [0.5, -0.3]  
- Target: [1.0]  
- Architecture: 2 → 4 (hidden) → 1 (output)  

**Sample Output**  
```  
Epoch 0, Loss: 1.2345  
Epoch 100, Loss: 0.8765  
...  
Epoch 900, Loss: 0.0123  
```  

**Customization**  
Modify these parameters in `main()`:  
- Network architecture (layer sizes)  
- Learning rate (`learning_rate` variable)  
- Training epochs  
- Input/target data  

**Limitations**  
- Single-threaded CPU implementation  
- Basic optimization (SGD only)  
- No batch processing support
