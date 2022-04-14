# Neural Network Interpretation

Để có thể đưa ra dự đoán, dữ liệu đầu vào được truyền thông qua nhiều lớp (layers) bao gồm các phép nhân ma trận giữ dữ
liệu và các trọng số mà model đã học được (learned weights) và thông qua các phép biến đổi phi tuyến tính (non-linear 
transformations)

Để có thể hiểu được hành vi mà model đưa ra quyết định, ta cần có phương pháp để hiểu model:
1. Mạng Neural Net học các features và concepts bên trong những lớp ẩn của nó và ta cần một tool đặc biệt để có thể xem xét
chúng
2. Tín hiệu gradient truyền ngược có thể sử dụng để hiểu model. 

Các kỹ thuật được áp dụng như sau:
* [Learned Features](#learned-features): Những features nào mà network đã học được ?
* [Pixel Attribution (Saliency Maps)](#pixel-attribution-saliency-maps): Các pixel đóng góp như thế nào vào việc đưa ra quyết định của model ?
* [Concepts](): Những concept trừu tượng nào mà Neural Net đã học ? 
* [Adversarial Examples](): Làm cách nào ta có thể đánh lừa được Neural Network
* [Influential Instances](): ???


## Learned Features 

Với CNN, ảnh được đưa vào mạng ở dạng raw (pixels). Mạng sẽ biến đổi hình ảnh này nhiều lần. Đầu tiên, ảnh sẽ thông qua 
nhiều lớp convolutional layers. Bên trong các lớp đó, mạng sẽ học được nhiều thứ và bắt đầu tăng dần mức độ phức tạp của 
mà nó học được. Cuối cùng thông tin hình ảnh sau khi biến đổi bởi model sẽ thông qua lớp Fully connected layers và 
sẽ chuyển thành dự đoán. 

* Những tầng đầu tiên của CNN học những đặc trưng như là cạnh hoặc là các mẫu đơn giản.
* Những tầng sâu hơn học những mẫu phức tạp hơn.
* Tầng sâu nhất của CNN học các đặc trưng như là toàn bộ hay là một phần của vật thể.
* Tầng fully connected layers học cách kết nối các activations từ các features nhiều chiều về các lớp cần được phân tách.

Các cách để có thể chứng minh cho những điều nói ở trên

### Feature Visualization

Đây là cách tiếp cận trực tiếp. Việc trực quan các đặc trưng tại một đơn vị của Neural Network được hiểu là tìm một 
đầu vào sao cho nó làm đơn vị đó cho ra mức độ activation cao nhất. 

Đơn vị (unit) ở đây được hiểu là các neuron đơn lẻ, các kênh (channels hoặc là feature maps), toàn bộ các lớp của tầng 
cuối của bộ Classification (hoặc tương đương là bộ feature trước khi qua pre-softmax để map sang probability).

Các neuron là đơn vị của mạng lưới, vì thế ta có thể có được thông tin nhiều nhất khi tạo bộ feature visualizations cho 
từng neuron. Nhưng có một vấn đề xảy ra: Neural network thông thường bao gồm hàng triệu neurons. Việc nhìn vào từng neuron 
sẽ tốn thời gian. Channel (thường gọi là activation maps - là kết quả của việc slide 1 Neuron trên toàn bộ input) là một 
lựa chọn tốt hơn cho feature visualization. Ta có thể tiến thêm một mức xa hơn đó là visualize toàn bộ lớp convolutional

#### Feature Visualization through Optimization

Feature visualization là bài toán về tối ưu. Ta giả thiết là toàn bộ trọng số trong neural network được fix cố định, 
điều đó có nghĩa là network đã được huấn luyện trước đó rồi. Ta sẽ tìm tấm ảnh cho phép maximize activation của một đơn
vị.

Đối với việc tối ưu ảnh sao cho maximize output của một đơn vị neuron, ta có công thức: 

![](images/im_1.png)

Hàm h chính là activation của neuron, img là input của mạng (hình) ảnh, x, y mô tả vị trí mà neuron đang thực hiện tích 
chập, n là chỉ số của layer và z là chỉ số của channel. 

Đối với trường hợp mục tiêu cần tối ưu là toàn bộ channel z trong layer n, thì ta sẽ tối ưu activation: 

```text
Bài toán tối ưu trên có thể hiểu như sau:
* Mục tiêu là tối ưu (maximize) output của các đơn vị mà ta muốn visual
* Để làm được như vậy thì ta sẽ điều chỉnh ảnh đầu vào dần dần sao cho output của các đơn vị đó là lớn nhất
* Ở đây tham số cần điều chỉnh đó chính là giá trị của từng pixel trong ảnh kết quả. 
* Để có thể điều chỉnh thì ta sẽ tính toán đạo hàm từng phần của gía trị Loss w.r.t theo các giá trị pixel đó
* Pytorch sử dụng cơ chế Autograd để có thể thực hiện được việc đó.
```

![](images/im_2.png)

Trong công thức trên, toàn bộ neuron đều có trọng số như nhau. Ngoài ra ta có thể random chọn một bộ số ngẫu nhiên theo 
các hướng khác nhau, bao gồm cả hướng âm. Thay vì maximize activation, ta có thể minimize chúng (điều này tương tụ với 
maximizing theo hướng âm)

Có 2 hướng để tạo ra ảnh input:

* Cách đầu tiên là sử dụng lại ảnh trong dữ liệu train, tìm những tấm cho phép maximize activation. Đây là cách tiếp cận hợp lệ,
nhưng sử dụng dữ liệu huấn luyện có một vấn đề là các phần tử trong hình có thể tương quan với nhau và chúng ta không thể
thấy mạng neuron đang thực sự tìm kiếm điều gì.

* Cách thứ hai là tạo ra một ảnh mới, bắt đầu từ một random noise. Để có thể tạo được một hình ảnh minh họa có ý nghĩa, 
ta phải thêm một vài yêu cầu: chỉ những thay đổi nhỏ mới được phép. Để có thể giảm thiểu được noise trong ảnh, ta có thể 
áp dụng jittering, rotation hoặc là scale với ảnh trước khi áp dụng optimization. 

### Connection to Adversarial Examples 

Có sự liên quan giữa feature visualization và adversarial. Cả hai kỹ thuật đều cực đại kích hoạt của đơn vị neuron. Đối 
với adversarial, ta tìm cách maximize activation cho lớp đối thủ (incorrect) class. Một điểm khác nữa đó là khởi tạo ban
đầu: với Adversarial, đó là tấm ảnh chúng ta muốn làm fool model. Đối với feature visualization, nó phụ thuộc vào cách 
tiếp cận, có thể là random noise. 


### Implementation, Result and Evaluation
#### 1. Implementation
Code của cái này được mình code dựa vào kiến thức đã có ở trên, và mình sử dụng **Pytorch Framework** để cài đặt quá trình 
Optimization ảnh đầu vào để được ảnh mô tả feature mà đơn vị của mạng đã được huẩn luyện từ trước học được (khái niệm 
học được ở đây có nghĩa là tín hiệu sẽ cho ra cao - đối với những đầu vào mà đơn vị đó đã học được để nhận biết mẫu 
hoặc bộ phận hay họa tiết ...)

#### 2. Usage

Code của mình đã cho phép visualize các loại features dựa vào các tiêu chí mà ta muốn tối ưu: Neuron (Đang cài đặt), 
Channel (Một feature map), Class Logits (Các neurons ở lớp Pre Softmax) - (Còn Layer (DeepDream) và Class Probability là 
chưa viết). 

Ta có thể sử dụng bất kỳ model nào từ package ```torchvision.models```

Code sẽ được chạy từ script ```main.py```. 

Ví dụ mẫu, ta áp dụng Model VGG16 có kiến trúc như sau:

```
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace=True)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace=True)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace=True)
    (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): ReLU(inplace=True)
    (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace=True)
    (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (20): ReLU(inplace=True)
    (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (22): ReLU(inplace=True)
    (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): ReLU(inplace=True)
    (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (27): ReLU(inplace=True)
    (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (29): ReLU(inplace=True)
    (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace=True)
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace=True)
    (5): Dropout(p=0.5, inplace=False)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
```

Nếu ta muốn visual cho đơn vị là Channel tại Filter thứ 10 của Layer Conv2d (28) của model thì ta sẽ set các tham số như sau:
```
layer = "features.28"
filter = 9
```

Nếu ta muốn visual cho đơn vị là Neuron đại diện cho Class 1 (goldfish) của ImageNet tại lớp Pre Softmax thì ta sẽ thực
hiện điều chỉnh tham số như sau (assume ta đang sử dụng model VGG16)

``` 
layer = "classifier.6"
filter = 1
```

#### 3. Results

##### 3.1. Channel objective (Channel as Unit)

Ở đây mình sử dụng model VGG16 để visualize, cách gọi đơn vị phụ thuộc vào cái mô hình model trong Pytorch

| Đơn vị         | Ảnh Feature tại đó                  |
|----------------|-------------------------------------|
| Layer(0)[63]   | ![](images/layer_0_filter_63.jpg)   |
| Layer(12)[3]   | ![](images/layer_12_filter_3.jpg)   |
| Layer(12)[105] | ![](images/layer_12_filter_105.jpg) |
| Layer(12)[181] | ![](images/layer_12_filter_181.jpg) |
| Layer(12)[231] | ![](images/layer_12_filter_231.jpg) |
| Layer(14)[181] | ![](images/layer_14_filter_181.jpg) |
| Layer(19)[54]  | ![](images/layer_19_filter_54.jpg)  |
| Layer(21)[54]  | ![](images/layer_21_filter_54.jpg)  |
| Layer(24)[54]  | ![](images/layer_24_filter_54.jpg)  |
| Layer(26)[1]   | ![](images/layer_26_filter_1.jpg)   |
| Layer(26)[54]  | ![](images/layer_26_filter_54.jpg)  |
| Layer(26)[200] | ![](images/layer_26_filter_200.jpg) |
| Layer(28)[54]  | ![](images/layer_28_filter_54.jpg)  |
| Layer(28)[264] | ![](images/layer_28_filter_264.jpg) |
| Layer(28)[265] | ![](images/layer_28_filter_265.jpg) |
| Layer(28)[266] | ![](images/layer_28_filter_266.jpg) |
| Layer(28)[462] | ![](images/layer_28_filter_462.jpg) |

##### 3.2. Class Logits objective (Class Logits as Unit)

Ở đây mình sử dụng model VGG19 để visualize: (Cái này để sau mình đẩy ảnh lên, mất tiêu cmnr)

| Đơn vị                 | Class trong ImageNet                                                           | Ảnh Feature tại đây                   |
|------------------------|--------------------------------------------------------------------------------|---------------------------------------|
| Layer(classifier)[1]   | goldfish, Carassius auratus                                                    | ![](images/layer_classifier_61.jpg)   |
| Layer(classifier)[3]   | tiger shark, Galeocerdo cuvieri                                                | ![](images/layer_classifier_63.jpg)   |
| Layer(classifier)[7]   | cock                                                                           | ![](images/layer_classifier_67.jpg)   |
| Layer(classifier)[30]  | bullfrog, Rana catesbeiana                                                     | ![](images/layer_classifier_630.jpg)  |
| Layer(classifier)[37]  | box turtle, box tortoise                                                       | ![](images/layer_classifier_637.jpg)  |
| Layer(classifier)[48]  | Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis | ![](images/layer_classifier_648.jpg)  |
| Layer(classifier)[49]  | African crocodile, Nile crocodile, Crocodylus niloticus                        | ![](images/layer_classifier_649.jpg)  |
| Layer(classifier)[71]  | scorpion                                                                       | ![](images/layer_classifier_671.jpg)  |
| Layer(classifier)[76]  | tarantula                                                                      | ![](images/layer_classifier_676.jpg)  |
| Layer(classifier)[79]  | centipede                                                                      | ![](images/layer_classifier_679.jpg)  |
| Layer(classifier)[92]  | bee eater                                                                      | ![](images/layer_classifier_692.jpg)  |
| Layer(classifier)[99]  | goose                                                                          | ![](images/layer_classifier_699.jpg)  |
| Layer(classifier)[105] | koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus          | ![](images/layer_classifier_6105.jpg) |
| Layer(classifier)[107] | jellyfish                                                                      | ![](images/layer_classifier_6107.jpg) |
| Layer(classifier)[285] | Egyptian cat                                                                   | ![](images/layer_classifier_6285.jpg) |
| Layer(classifier)[301] | ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle                    | ![](images/layer_classifier_6301.jpg) |
| Layer(classifier)[309] | bee                                                                            | ![](images/layer_classifier_6309.jpg) |
| Layer(classifier)[316] | cicada, cicala                                                                 | ![](images/layer_classifier_6316.jpg) |
| Layer(classifier)[403] | aircraft carrier, carrier, flattop, attack aircraft carrier                    | ![](images/layer_classifier_6403.jpg) |
| Layer(classifier)[417] | balloon                                                                        | ![](images/layer_classifier_6417.jpg) |
| Layer(classifier)[445] | bikini, two-piece                                                              | ![](images/layer_classifier_6445.jpg) |
| Layer(classifier)[485] | CD player                                                                      | ![](images/layer_classifier_6485.jpg) |
| Layer(classifier)[488] | chain                                                                          | ![](images/layer_classifier_6488.jpg) |
| Layer(classifier)[492] | chest                                                                          | ![](images/layer_classifier_6492.jpg) |
| Layer(classifier)[508] | computer keyboard, keypad                                                      | ![](images/layer_classifier_6508.jpg) |
| Layer(classifier)[531] | digital watch                                                                  | ![](images/layer_classifier_6531.jpg) |
| Layer(classifier)[562] | fountain                                                                       | ![](images/layer_classifier_6562.jpg) |
| Layer(classifier)[605] | iPod                                                                           | ![](images/layer_classifier_6605.jpg) |
| Layer(classifier)[629] | lipstick, lip rouge                                                            | ![](images/layer_classifier_6629.jpg) |
| Layer(classifier)[657] | missile                                                                        | ![](images/layer_classifier_6657.jpg) |
| Layer(classifier)[805] | soccer ball                                                                    | ![](images/layer_classifier_6805.jpg) |
| Layer(classifier)[879] | umbrella                                                                       | ![](images/layer_classifier_6879.jpg) |
| Layer(classifier)[971] | bubble                                                                         | ![](images/layer_classifier_6971.jpg) |
| Layer(classifier)[987] | corn                                                                           | ![](images/layer_classifier_6987.jpg) |

#### 4. Evaluation

Để kiểm tra tính đúng đắn cho quá trình Optimization, ta thử feed ảnh được Optimized sẵn cho mỗi class của VGG19 ở trên 
vào trong script ```main.py``` của thư mục ```CheckingModelPrediction```:

Đối với class: umbrela
Ta feed vào model ảnh: 
![](./images/layer_classifier_6879.jpg)
```python
Prediction of VGG model 
Class: umbrella 
Percentage: 99.99665069580078
```

Đối với class: goldFish 
Ta feed vào model ảnh:
![](./images/layer_classifier_61.jpg)
```python 
Prediction of VGG model 
Class: goldfish, Carassius auratus 
Percentage: 100.0
```

### Network Dissection

Phần này chưa đụng tới. 


## Pixel Attribution (Saliency Maps)

Pixel attribution methods (phương pháp liên quan đến sự phân bổ của pixel) làm nổi bật các pixel trong ảnh đầu vào có 
liên quan đến một lớp phân loại ảnh nhất định trong mạng thần kinh. 

Pixel attribution methods còn có nhiều tên gọi khác nhau: Sensitivity map, saliency map, pixel attribution map, gradient-based 
attribution methods, feature relevance, feature attribution và feature contribution.

Pixel attribution là một trường hợp đặc biệt của feature attribution, áp dụng đối với ảnh. Feature attribution giải thích
các dự đoán riêng lẻ bằng cách phân bổ từng tính năng vào (input feature) theo mức độ mà nó đã thay đổi dự đoán (tiêu cực hoặc tích 
cực). Features ở đây có thể là input pixels, bảng biểu hoặc là các từ ...

Có 2 kiểu của phương pháp attribution:

* **Occlusion-or pertubation-based**: Cái này chưa đụng tới
* **Gradient-based**: Rất nhiều phương pháp tính toán gradient của prediction (hoặc là classification score) w.r.t input
features. Các phương pháp liên quan đến Gradient-based đa phần khác nhau ở cách mà gradient được tính toán. 

Cả hai cách tiếp cận này đều có điểm chung đó là kết quả tạo ra có cùng kích thước đối với lại ảnh đầu vào, và chúng gán cho mỗi pixel
một giá trị có thể được hiểu là mức độ liên quan của pixel đối với dự đoán hoặc phân loại của hình ảnh đó. 


Ta sử dụng các ký hiệu như sau, ta có một mạng neuron có đầu ra dự đoán là một vector có chiều dài là C. Đầu ra từ mạng cho 
ảnh đầu vào I được gọi là S(I) và: 

<img src="https://render.githubusercontent.com/render/math?math=S(I) = [S_1(I), S_2(I), ..., S_C(I)]">

Tất cả các phương pháp nhận đầu vào là <img src="https://render.githubusercontent.com/render/math?math=x \in \mathbb{R} ^ p"> 
(có thể là image pixels, tabular data, words, ...) với p features và đầu ra của thuật toán là sự giải thích cho các 
mức độ đóng góp của từng điểm đầu vào cho dự đoán: <img src="https://render.githubusercontent.com/render/math?math=R^c = [R_1^c, R_2^c, ..., R_p^c]">. 
Chỉ số c thể hiện cho class mà ta muốn visualize.

### Vanilla Gradient (Saliency Maps)

Ý tưởng về Vanilla Gradient được giới thiệu bởi Simonyan vào năm 2013. Đối với phương pháp này, ta sẽ tính toán gradient 
của hàm mất mát đối với lớp mà ta muốn visual w.r.t input pixels. Việc này tạo ra một map với kích thước tương đương với 
input features với các giá trị từ âm đến dương. 

Cách thực hiện của phương pháp này đó là: 
1. Thực hiện forward pass đối với ảnh mà ta muốn thực hiện
2. Tính toán gradient của lớp muốn visualize w.r.t với lại input pixels: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://render.githubusercontent.com/render/math?math=E_{grad}(I_0) = \frac{\partial S_c}{\partial I}|_{I = I_0}">

Ở đây ta thiết lập các class khác về 0
3. Visualize gradients. Ta có thể visual giá trị tuyệt đối hoặc highlight giá trị gradient âm lẫn dương.

Nói cách khác, ta có ảnh đầu vào I và mạng neuron tạo ra Score S_c(I) cho lớp c. Đó là kết quả của một hàm phi tuyến khi
áp dụng vào hình .ảnh Ý tưởng đằng sau việc sử dụng gradient đó là ta có thể xấp xỉ điểm số đó bằng cách sử dụng khai 
triển Taylor bậc 1: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://render.githubusercontent.com/render/math?math=S_c(I) \approx w^T I + b">

Với w là đạo hàm của điểm: 

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://render.githubusercontent.com/render/math?math=w = \frac{\partial S_c}{\partial I}|_{I_0}">

Ở đây, có một sự nhập nhằng khi thực hiện backward pass cho gradients, bởi vì các hàm phi tuyến như là ReLU. Vì thế khi 
thực hiện backwardpass, ta không biết gán giá trị dương hay âm cho activation. Theo ký hiệu, hàm ReLU áp dụng tại layer 
n + 1: 
```X_n+1(x) = max(0, X_n)``` từ lớp X_n sang lớp X_n+1. Điều này có nghĩa là khi activation của một neuron là 0, ta không
biết được giá trị nào nên backpropagate lại. Đối với trường hợp của Vanilla Gradient, sự nhập nhằng được giải quyết như sau:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial f}{\partial X_n} = \frac{\partial f}{\partial X_{n+1}}.I(X_n > 0)">

Ở đây, I là hàm element wise indicator function, giá trị tại các phần từ của nó là 0 nếu như activation tại vị trí tương 
ứng của lớp phía sau là âm, và là 1 nếu như giá trị tại activation là 0 hoặc là dương. Điều này tương ứng cho autograd của 
nn.ReLU trong Pytorch. 

Một ví dụ, ta có layers X_n và X_n+1 = ReLU(X_n)

Giả sử tại X_n ta có activation của nó là:

```
|1     0|
|-1  -10|
```

Và Gradient của ta tại Layer X_n+1 là: 
```
|0.4     1.1|
|-0.5   -0.1|
```

Thì gradients của ta tại X_n là:

```
|0.4    0|
|0      0|
```

## References
[1] [Feature Visualization Implementation using FastAI](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030)

[2] [Feature Visualization Distill pub](https://distill.pub/2017/feature-visualization/)

[3] [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)



