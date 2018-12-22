# Classification-convNet

## Description
In this project, we implement a neural network to predict the class labels of a given image without using any deep learning packages.

## Dataset
-  This dataset contains 10,000 images in total, which are divided into 20 classes (each class has 500 images). The class labels are as follows.

</ul>
<table id="lectures">
<tr class="r1"><td class="c1">label index  </td><td class="c2"><b>1</b>       </td><td class="c3"><b>2</b>     </td><td class="c4"><b>3</b>    </td><td class="c5"><b>4</b>  </td><td class="c6"><b>5</b></td><td class="c7">  <b>6</b> </td><td class="c8">  <b>7</b></td><td class="c9">  <b>8</b></td><td class="c10">  <b>9</b></td><td class="c11">  <b>10</b></td><td class="c12">  <b>11</b></td><td class="c13">  <b>12</b></td><td class="c14">  <b>13</b></td><td class="c15">  <b>14</b></td><td class="c16">  <b>15</b></td><td class="c17">  <b>16</b></td><td class="c18">  <b>17</b></td><td class="c19">  <b>18</b></td><td class="c20">  <b>19</b></td><td class="c21">  <b>20</b> </td></tr>
<tr class="r2"><td class="c1">text description </td><td class="c2"> goldfish </td><td class="c3"> frog </td><td class="c4"> koala </td><td class="c5"> jellyfish </td><td class="c6"> penguin </td><td class="c7"> dog </td><td class="c8"> yak </td><td class="c9"> house </td><td class="c10"> bucket </td><td class="c11"> instrument </td><td class="c12"> nail </td><td class="c13"> fence </td><td class="c14"> cauliflower </td><td class="c15"> bell peper </td><td class="c16"> mushroom </td><td class="c17"> orange </td><td class="c18"> lemon </td><td class="c19"> banana </td><td class="c20"> coffee     </td><td class="c21">    beach            
</td></tr></table>
<p><br /></p>
<ul>


- The dataset is cultured from [the tiny ImageNet dataset](https://tiny-imagenet.herokuapp.com/). You may want to visit the source page for more details
- Most of training images are color images, and each of them has 64 * 64 pixels. The others are grey images, and each of them has 64 * 64 pixels as well.

##The Classifier
- Input: an image.

- Output: the confidence scores of all classes. If the input image is belong to one of the 20 classes, the confidence score corresponding to its true class is expected to be the largest one. Otherwise, the confidence score of a label “unknown” is supposed to be the largest one. Thus, you indeed have 21 class labels.

##The Performance Measure
- For each testing image, your classifier will output the top three class labels with the highest confidence scores. If the true class label is one of the top three labels, we will say your classifier has correctly predicted the class label of the input image; otherwise, your classifier made a mistake.

## Training

## Testing
