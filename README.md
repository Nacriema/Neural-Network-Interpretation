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
  * [Vanilla Gradient](#vanilla-gradient-saliency-maps)
  * [DeconvNet](#deconvnet)
  * [Grad-CAM](#grad-cam)
  * [Guided Grad-CAM](#guided-grad-cam)
  * [SmoothGrad](#smoothgrad)
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

Đối với class: umbrela , ta feed vào model ảnh: 
![](./images/layer_classifier_6879.jpg)
```python
Prediction of VGG model 
Class: umbrella 
Percentage: 99.99665069580078
```

Đối với class: goldFish, ta feed vào model ảnh:
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

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://render.githubusercontent.com/render/math?math=S_c(I) \approx w^T I plus b">

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

#### Vấn đề của Vanilla Gradient
Vanilla Gradient bị vấn đề đó là việc bão hòa, được đề cập bởi [Avanti et al. 2017](https://arxiv.org/abs/1704.02685). Khi 
sử dụng ReLU, và khi activation xuống dưới ngưỡng 0, thì activation sẽ bị giới hạn ở 0 và không thay đổi nữa. Việc activation 
sẽ bị bão hòa. Ví dụ: Đầu vào của một layer là 2 neuron với trọng số kết nối cho từng cái là -1 và -1 và một bias là 1.

Hàm kích hoạt sẽ là neuron1 + neuron2 (a + b) nếu như tổng của 2 neuron là < 1 (tức là khi -a-b+1>0 <=> a+b<1 thì activation
sẽ là -a-b+1). Ngược lại nếu tổng của 2 neuron mà lớn hơn 1 thì activation sẽ luôn bị bão hòa tại activation của 1. Và gradient tại 
điểm này cũng sẽ về 0, và theo Vanilla Gradient thì neuron này sẽ không quan trọng. 

```
Output của neuron đó là: 
output = a * (-1) + b * (-1) + 1 

Thông qua ReLU: max(0, output)
```


#### Implementation, Result and Evaluation

##### 1. Implementation
##### 2. Result

Kết quả thu được khi áp dụng Vanilla Gradient đối với VGG19

| Ảnh gốc                             | Colored Gradient                                     | Negative Gradient                                            | Positive Gradient                                            | Heat Map                                               | Image overlay heat map                                  | 
|-------------------------------------|------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------|---------------------------------------------------------|
| ![](images/cat_dog.png)             | ![](images/cat_dog_Vanilla_BP_color.jpg)             | ![](images/cat_dog_Vanilla_BP_negative_grad.jpg)             | ![](images/cat_dog_Vanilla_BP_positive_grad.jpg)             | ![](images/cat_dog_Vanilla_BP_Heatmap.png)             | ![](images/cat_dog_Vanilla_BP_On_Image.png)             |
| ![](images/goldfinch.png)           | ![](images/goldfinch_Vanilla_BP_color.jpg)           | ![](images/goldfinch_Vanilla_BP_negative_grad.jpg)           | ![](images/goldfinch_Vanilla_BP_positive_grad.jpg)           | ![](images/goldfinch_Vanilla_BP_Heatmap.png)           | ![](images/goldfinch_Vanilla_BP_On_Image.png)           |
| ![](images/hay2.jpeg)               | ![](images/hay2_Vanilla_BP_color.jpg)                | ![](images/hay2_Vanilla_BP_negative_grad.jpg)                | ![](images/hay2_Vanilla_BP_positive_grad.jpg)                | ![](images/hay2_Vanilla_BP_Heatmap.png)                | ![](images/hay2_Vanilla_BP_On_Image.png)                |
| ![](images/house_finch.png)         | ![](images/house_finch_Vanilla_BP_color.jpg)         | ![](images/house_finch_Vanilla_BP_negative_grad.jpg)         | ![](images/house_finch_Vanilla_BP_positive_grad.jpg)         | ![](images/house_finch_Vanilla_BP_Heatmap.png)         | ![](images/house_finch_Vanilla_BP_On_Image.png)         |
| ![](images/killer_whale.png)        | ![](images/killer_whale_Vanilla_BP_color.jpg)        | ![](images/killer_whale_Vanilla_BP_negative_grad.jpg)        | ![](images/killer_whale_Vanilla_BP_positive_grad.jpg)        | ![](images/killer_whale_Vanilla_BP_Heatmap.png)        | ![](images/killer_whale_Vanilla_BP_On_Image.png)        |
| ![](images/rubber_eraser.png)       | ![](images/rubber_eraser_Vanilla_BP_color.jpg)       | ![](images/rubber_eraser_Vanilla_BP_negative_grad.jpg)       | ![](images/rubber_eraser_Vanilla_BP_positive_grad.jpg)       | ![](images/rubber_eraser_Vanilla_BP_Heatmap.png)       | ![](images/rubber_eraser_Vanilla_BP_On_Image.png)       |
| ![](images/snake.jpg)               | ![](images/snake_Vanilla_BP_color.jpg)               | ![](images/snake_Vanilla_BP_negative_grad.jpg)               | ![](images/snake_Vanilla_BP_positive_grad.jpg)               | ![](images/snake_Vanilla_BP_Heatmap.png)               | ![](images/snake_Vanilla_BP_On_Image.png)               |
| ![](images/spider.png)              | ![](images/spider_Vanilla_BP_color.jpg)              | ![](images/spider_Vanilla_BP_negative_grad.jpg)              | ![](images/spider_Vanilla_BP_positive_grad.jpg)              | ![](images/spider_Vanilla_BP_Heatmap.png)              | ![](images/spider_Vanilla_BP_On_Image.png)              |
| ![](images/layer_classifier_61.jpg) | ![](images/layer_classifier_61_Vanilla_BP_color.jpg) | ![](images/layer_classifier_61_Vanilla_BP_negative_grad.jpg) | ![](images/layer_classifier_61_Vanilla_BP_positive_grad.jpg) | ![](images/layer_classifier_61_Vanilla_BP_Heatmap.png) | ![](images/layer_classifier_61_Vanilla_BP_On_Image.png) |

##### 3. Evaluation

### DeconvNet

DeconvNet được đề xuất bởi [Zeiler và Fergus (2014)](https://arxiv.org/abs/1409.1556), nó gần như tương đồng với Vanilla 
Gradient. Mục tiêu của DeconvNet đó là đảo ngược mạng neuron và bài báo đề xuất các toán tử nhằm mục đích đảo ngược các 
lớp lọc (filtering), lớp gộp (pooling) và các lớp kích hoạt (activation layers). Khi ta nhìn vào bài báo, nó sẽ trông rất 
khác đối với Vanilla Gradient, nhưng ngoài sự đảo ngược của lớp ReLU, DeconvNet tương đương với cách tiếp cận Vanilla 
Gradient. Vanilla Gradient có thể được xem là sự tổng quát của DeconvNet. DeconvNet đưa ra sự lựa chọn khác cho việc backpropagate 
gradient qua lớp ReLU:

```
            R_n = R_n+1 * 1 (R_n+1 > 0) 
```


Với R_n và R_n+1 là đạo hàm của hàm loss cho các tham số trong từng lớp tương ứng và 1 là indicator function. Khi backpass 
từ lớp n sang lớp n - 1, DeconvNet "ghi nhớ" những điểm activation nào trong lớp n đã được set bằng 0 ở giai đoạn forward pass  
và set chúng bằng 0 tại lớp n-1. Activation với giá trị âm ở lớp n sẽ được set bằng 0 trong lớp n - 1. Trong ví dụ ở trên, 
gradient tại lớp X_n đối với ví dụ ở trên sẽ là: 

```
|0.4        1.1|
|0            0|
```

#### Implementation, Result and Evaluation

##### 1. Implementation
##### 2. Result


##### 3. Evaluation


### Grad-CAM

Grad-CAM cung cấp cách thức để giải thích trực quan cho các quyết định của CNNs. Không giống như các phương pháp khác, 
gradient không phải được backpropagated ngược trở lại input mà thường là đến lớp CNN cuối cùng để tạo ra một bản đồ làm 
nổi bật các vùng quan trọng của hình ảnh.

Grad-CAM là viết tắt của Gradient-weighted Class Activation Map. Như cái tên gọi của nó, nó dựa vào gradient của mạng. 
Grad-CAM, cũng như các kỹ thuật khác gán mỗi một neuron một điểm số liên quan cho quyết định. Quyết định này có thể là 
một dự đoán cho lớp (class) mà ta tìm thấy được trong lớp đầu ra, **nhưng về mặt lý thuyết có thể là bất kỳ lớp nào khác 
trong mạng neuron.** Grad-CAM sao chép thông tin này vào lớp CNN cuối cùng. Grad-CAM có thể được sử dụng với các loại CNNs 
khác nhau: với Fully-connected layers, captioning hoặc là multi-task output, và phục vụ cho việc học tăng cường. 

Mục tiêu của Grad-CAM đó là hiểu được phần nào của ảnh mà một lớp Convolutional layer nhìn vào để đưa ra dự đoán cho một 
lớp cụ thể mà ta đang xét đến. Lớp CNN đầu tiên lấy đầu vào là hình ảnh và đầu ra là bản đồ các tính năng mã hóa mà lớp 
đó đã học được. Các lớp tích chập cao hơn cũng làm tương tự như vậy, nhưng lấy bản đồ tính năng cúa các lớp trước làm đầu 
vào cho lớp đó. Để hiểu cách CNN đưa ra quyết định, Grad-CAM phân tích vùng nào được kích hoạt trong bản đồ tính năng của 
các lớp tích chập cuối cùng. Giả sử ta có k feature maps trong lớp tích chập cuối cùng, ta gọi chúng lần lượt là A_1, A_2, ...
, A_k. Làm thế nào để ta thấy được từ feature maps cách mà CNN đưa ra được quyết định. 

Trong cách tiếp cận đầu tiên, ta đơn giản là visualize giá trị raw của mỗi một feature map, trung bình cộng chúng và sau 
đó đè chúng lên trên ảnh ban đầu của chúng ta. Điều này có thể sẽ không mấy tốt đẹp, bởi vì feature map mã hóa thông tin 
**cho tất cả các lớp**, nhưng chúng ta chỉ muốn một class thôi. Kết quả của ý tưởng này được minh họa bằng bảng kết quả trong 
phần **Result**



Chính vì lý do đó, công việc của Grad-CAM đó là quyết định mức độ quan trọng của từng k feature map đóng góp cho việc phân loại lớp c mà ta mong muốn quan sát. Ta cần phải gán trọng số 
cho mỗi pixel trong mỗi một feature map với lại gradient trước khi ta lấy trung bình cộng theo channel của feature maps.
Điều này sẽ mang lại cho ta một biểu đồ nhiệt highlight được các vùng mà có ảnh hưởng tích cực hoặc tiêu cực đến lớp phân
loại mà ta đang muốn xem xét. Biểu đồ này được gửi vào thông qua hàm ReLU, là một cách để set toàn bộ những giá trị âm về
0 , Grad-CAM loại bỏ những giá trị âm bằng cách sử dụng hàm ReLU, với lập luận rằng chúng ta chỉ quan tâm đến các phần đóng 
góp vào lớp c được chọn chứ không quan tâm đến các phần khác. Chữ pixel có thể gây hiểu nhầm bởi vì feature map được tạo ra 
là nhỏ hơn hình ảnh ban đầu (bởi vì các đơn vị pooling) nhưng được ánh xạ trở lại hình ảnh gốc. Sau đó ta scale Grad-CAM 
về lại khoảng [0, 1] cho mục đích visualize và phủ nó lên hình ảnh ban đầu. 

Công thức của **Grad-CAM** như sau: mục tiêu là tìm map L cho lớp c được định nghĩa như sau: 

<img src="https://render.githubusercontent.com/render/math?math=L^c_{Grad-CAM} \in \mathbb{R}^{u\times v} = ReLU(\sum_{k}\alpha^c_kA^k)">

Ở đây u là chiều rộng, v là chiều cao của Grad-CAM map và c là class mà ta muốn visualize. Các bước thực hiện như sau:

1. Forward pass ảnh vào thông qua mạng CNN
2. Lấy được score raw của lớp ta muốn visual (có nghĩa là score tại neuron đại diện cho class và ở lớp Pre-softmax)
3. Đặt các giá trị còn lại về 0
4. Thực hiện backpropagate từ lớp ưa thích đó đến trước lớp fully-connected layer: <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial y^c}{\partial A^k}"> 
5. Đánh trọng số các feature map "pixel" bằng gradient cho mỗi lớp. Chỉ số i, j bên dưới ký hiệu cho chiều rộng và chiều cao:

<img src="https://render.githubusercontent.com/render/math?math=\alpha^c_k = \frac{1}{Z}\sum_i \sum_j \frac{\partial y^c}{\partial A^k_{ij}}">

Điều này có nghĩa là gradient sẽ được tính trung bình cộng (Chia cho Z  = i*j)


6. Tính toán giá trị trung bình cho feature maps, trọng số trên mỗi pixel theo gradient.
7. Áp dụng ReLU vào average feature map
8. Để thực hiện visualization: Scale giá trị vào trong ngưỡng từ 0-1. Upscale heatmap lên và overlay nó lên ảnh nguyên bản
9. Bước tiến hành thêm đối với phương pháp Guided Grad-CAM: nhân heat map với guided backpropagation. 

#### Implementation, Result and Evaluation

##### 1. Implementation
##### 2. Result

| Ảnh gốc                             | Interested Class                          | Grad-CAM ý tưởng đơn giản                                               | Grad-CAM chuẩn                                            |
|-------------------------------------|-------------------------------------------|-------------------------------------------------------------------------|-----------------------------------------------------------|
| ![](images/cat_dog.png)             | Tibetan mastiff                           | ![](images/Grad-CAM_FirstApproach/cat_dog_Cam_On_Image.png)             | ![](images/Grad-CAM/cat_dog_Cam_On_Image.png)             |
| ![](images/goldfinch.png)           | goldfinch, Carduelis carduelis            | ![](images/Grad-CAM_FirstApproach/goldfinch_Cam_On_Image.png)           | ![](images/Grad-CAM/goldfinch_Cam_On_Image.png)           |
| ![](images/hay2.jpeg)               | hay                                       | ![](images/Grad-CAM_FirstApproach/hay2_Cam_On_Image.png)                | ![](images/Grad-CAM/hay2_Cam_On_Image.png)                |
| ![](images/house_finch.png)         | house finch, linnet, Carpodacus mexicanus | ![](images/Grad-CAM_FirstApproach/house_finch_Cam_On_Image.png)         | ![](images/Grad-CAM/house_finch_Cam_On_Image.png)         |
| ![](images/killer_whale.png)        | killer whale                              | ![](images/Grad-CAM_FirstApproach/killer_whale_Cam_On_Image.png)        | ![](images/Grad-CAM/killer_whale_Cam_On_Image.png)        |
| ![](images/rubber_eraser.png)       | rubber eraser                             | ![](images/Grad-CAM_FirstApproach/rubber_eraser_Cam_On_Image.png)       | ![](images/Grad-CAM/rubber_eraser_Cam_On_Image.png)       |
| ![](images/snake.jpg)               | snake                                     | ![](images/Grad-CAM_FirstApproach/snake_Cam_On_Image.png)               | ![](images/Grad-CAM/snake_Cam_On_Image.png)               |
| ![](images/spider.png)              | spider                                    | ![](images/Grad-CAM_FirstApproach/spider_Cam_On_Image.png)              | ![](images/Grad-CAM/spider_Cam_On_Image.png)              |
| ![](images/layer_classifier_61.jpg) | goldfish, Carassius auratus               | ![](images/Grad-CAM_FirstApproach/layer_classifier_61_Cam_On_Image.png) | ![](images/Grad-CAM/layer_classifier_61_Cam_On_Image.png) |


##### 3. Evaluation


### Guided Grad-CAM

Từ nội dung thuật toán của Grad-CAM, ta có thể thấy rằng heat map cho ra sẽ thô (bởi vì feature map thu được ỏ những lớp
cuối cùng của CNN thì kích thước ảnh sẽ nhỏ đi rất nhiều so với kích thước ảnh ban đầu, và ta phải resize heat map lên 
cho vừa với lại ảnh đầu vào). Ở chiều hướng ngược lại, các kỹ thuật khác sử dụng Gradient được lan truyền ngược đến các 
pixel ảnh đầu vào. Điều này tạo ra hình ảnh có độ chi tiết cao hơn nhiều và có thể hiển thị cho ta được các cạnh hoặc 
các điểm riêng lẻ nào đóng góp nhiều nhất vào dự đoán. Sự kết hợp của cả hai phương pháp được gọi là **Guided Grad-CAM**.
Việc tạo ra nó đơn giản. Ta tính toán cho cả Grad-CAM và kết hợp thêm sự giải thích từ một phương pháp khác, chẳng hạn 
như là Vanilla Gradient. Đầu ra của Grad-CAM sau đó sẽ được Upsample bởi **Bilinear Interpolation**, sau đó cả hai map 
này sẽ được nhân lại với nhau theo kiểu **element-wise**. Grad-CAM sẽ hoạt động giống như một công cụ tập trung vào các 
phần cụ thể của bản đồ phân bổ theo pixel.

#### Implementation, Result and Evaluation

##### 1. Implementation
##### 2. Result

| Ảnh gốc                             | Interested Class                          | Guided Grad-CAM   (Vanilla + GradCAM)                            | 
|-------------------------------------|-------------------------------------------|------------------------------------------------------------------|
| ![](images/cat_dog.png)             | Tibetan mastiff                           | ![](images/Guided_Grad-CAM/cat_dog_Cam_On_Image.png)             |
| ![](images/goldfinch.png)           | goldfinch, Carduelis carduelis            | ![](images/Guided_Grad-CAM/goldfinch_Cam_On_Image.png)           | 
| ![](images/hay2.jpeg)               | hay                                       | ![](images/Guided_Grad-CAM/hay2_Cam_On_Image.png)                | 
| ![](images/house_finch.png)         | house finch, linnet, Carpodacus mexicanus | ![](images/Guided_Grad-CAM/house_finch_Cam_On_Image.png)         | 
| ![](images/killer_whale.png)        | killer whale                              | ![](images/Guided_Grad-CAM/killer_whale_Cam_On_Image.png)        | 
| ![](images/rubber_eraser.png)       | rubber eraser                             | ![](images/Guided_Grad-CAM/rubber_eraser_Cam_On_Image.png)       | 
| ![](images/snake.jpg)               | snake                                     | ![](images/Guided_Grad-CAM/snake_Cam_On_Image.png)               | 
| ![](images/spider.png)              | spider                                    | ![](images/Guided_Grad-CAM/spider_Cam_On_Image.png)              | 
| ![](images/layer_classifier_61.jpg) | goldfish, Carassius auratus               | ![](images/Guided_Grad-CAM/layer_classifier_61_Cam_On_Image.png) | 

##### 3. Evaluation

### SmoothGrad

Ý tưởng của SmoothGrad được đề xuất bởi [Smilkov et al](https://arxiv.org/abs/1706.03825) nhằm mục đích giảm nhiễu cho
các phương pháp dựa vào gradient-base bằng cách thêm nhiễu và tính trung bình trên các gradient giả này. SmoothGrad không
phải là một phương pháp độc lập, nhưng nó có thể được áp dụng 


## References
[1] [Feature Visualization Implementation using FastAI](https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030)

[2] [Feature Visualization Distill pub](https://distill.pub/2017/feature-visualization/)

[3] [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)



