# Cat-and-dog
使用AlexNet识别猫和狗
使用的是tensorflow框架，项目里包括了若干剪裁过的图像，完整数据集在：
链接：https://pan.baidu.com/s/1m0i9hlqH0o_xXXJGH6Vn5Q 
提取码：h2v7 

图片结构应该是：
images--
        train --
                cats --
                      cat.001.jpg
                      ... ...
                dogs --
                      dog.001.jpg
                      ... ...
        test --
                cats --
                      cat.001.jpg
                      ... ...
                dogs --
                      dog.001.jpg
                      ... ...
        val --
                cats --
                      cat.001.jpg
                      ... ...
                dogs --
                      dog.001.jpg
                      ... ...
请务必注意图片文件夹名称和图片名称格式，代码中用到了文件夹名称来提取所属类别
