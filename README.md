# AI Image Processing and Classification Project

---

## AI Explanation

While I have minimal experience with Python, many of the AI's explanations for each line of code did make sense to me. This may be because I already have experience implementing image classification in Kotlin for my midterm app development project. For the image classification I implemented in that project, I used an API, and took similar steps in taking the image, resizing it, converting it to a bitmap (though in this instance, it is converting it into a NumPy array), and retrieving JSON data for best guesses for categories of food and parsing and using that data. 

In base_classifier.py, it seems that something similar may be happening based on the AI's explanation: the image array is fed to the model and then the predictions it gets are decoded and then displayed in the terminal.

If I had not already had previous experience implementing image classification, this would likely make way less sense to me than it does now. 
The only things that don't entirely make sense to me yet are the functions and syntax used. However, these are things I can pick up on quickly, and already being familiar with implementing image classification places me ahead of the game, I believe.

## Top 3 Predictions: basic_cat.jpg

- Top-3 Predictions:
- 1: tiger_cat (0.35)
- 2: tabby (0.34)
- 3: Egyptian_cat (0.07)

## Top 3 Predictions: elephant.jpg

- Top-3 Predictions:
- 1: Indian_elephant (0.30)
- 2: triceratops (0.30)
- 3: tusker (0.16)

## Heatmap Analysis

Upon running the Grad-CAM heatmap function on the elephant image and analyzing the saved Grad-CAM plot image, it was apparent which parts of the photo the model focused on the most when making its prediction.

Everything that has been highlighted in **red** or **hot pink** are the things that the model focused on the most. Those things are:
- The top of the elephant's head and trunk 
- The tips or edges of the elephant's ears
- Just below the elephant's ears
- The area around the elephant's trunk and mouth
- The elephant's rear
- Some of the greenery being held in the elephant's trunk

Everything that has been highlighted in **yellow** and **green** were focused on with medium importance/relevance. Those things are:
- The elephant's back
- The surface of the elephant's head where it meets its back
- The area surrounding the section of the trunk that is highlighted red
- Some of the greenery surrounding the elephant

Everything that has been highlighted or tinted **blue** were either not focused on at all or only had low or minimal relevance when predicting. Those things are:
- The general background of greenery
- The elephant's tusks
- The very center of the elephant's face
- A portion of the elephant's side

It is clear in order to make the prediction of **Indian elephant** from this image, the model focused heavily on the elephant's anatomy--the head, ears, rear, face, trunk--and some on the general context of the elephant's figure and the greenery surrounding or being held by it. It disregarded most of the background and some of the surface of the elephant's body, opting to focus on details like body and facial structure to make its guess.