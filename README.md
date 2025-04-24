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

---

# Occlusion Predictions & Confidence

I ran the three occlusion functions twice on the image of the elephant--once with a 50x50 sized box, then once more with a 100x100 sized box.

## Regular Image Classification
- Top-3 Predictions:
- 1: Indian_elephant (0.30)
- 2: triceratops (0.30)
- 3: tusker (0.16)

## 50x50 Results

**Occlusion: Black Patch**
- Top-3 Predictions (with black patch at (60,60)): 
- 1: Indian_elephant (0.15)
- 2: triceratops (0.09)
- 3: Komodo_dragon (0.04)

**Occlusion: Blur Patch**
- Top-3 Predictions (with blur patch at (60,60)):
- 1: triceratops (0.19)
- 2: Indian_elephant (0.17)
- 3: Komodo_dragon (0.07)

**Occlusion: Noise Patch**
- Top-3 Predictions (with noise patch at (60,60)):
- 1: Indian_elephant (0.25)
- 2: triceratops (0.16)
- 3: teddy (0.07)

## 100x100 Results

**Occlusion: Black Patch**
- Top-3 Predictions (with black patch at (60,60)): 
- 1: cliff_dwelling (0.05)
- 2: monitor (0.05)
- 3: screen (0.05)

**Occlusion: Blur Patch**
- Top-3 Predictions (with blur patch at (60,60)): 
- 1: earthstar (0.10)
- 2: lynx (0.06)
- 3: cougar (0.04)

**Occlusion: Noise Patch**
- Top-3 Predictions (with noise patch at (60,60)): 
- 1: jackfruit (0.22)
- 2: apron (0.08)
- 3: brain_coral (0.08)

# Occlusion Results Analysis

## Did the classifier struggle to classify the occluded images?

Yes, the classifier did struggle more to classify the occluded images, more-so with the larger occlusion patch. 

When using a 50x50 sized occlusion patch, the classifier correctly guessed Indian_elephant 2/3 times, triceratops the other 1/3. The confidence at which these guesses were made was also significantly lower than they were without the occlusion patches, going from (0.30) confidence to (0.25), (0.15), or (0.19) for the #1 predictions, with an even larger discrepancy between confidence scores on the 2nd and 3rd place predictions.

When using a 100x100 sized occlusion patch, the classifier's predictions were way off. Instead of guessing animals like elephant, triceratops, or komodo dragon, it guessed items like monitor, screen, earthstar, jackfruit, and brain_coral, or it guessed scenes like cliff_dwelling. The animals it did guess were cats like cougar and lynx, which are far from being an elephant.

## Which occlusion had the greatest impact on performance?

Just looking at the confidence scores, the black patch was the most disruptive and resulted in the lowest scores compared to image classification without occlusion.

However, looking at the actual predicted items, the occlusion that had the greatest impact on the model's performance was the blur patch. The 50x50 blur patch resulted in new predictions like Komodo_dragon and also was the only occlusion to result in the #1 prediction being a triceratops instead of Indian_elephant.

**Conclusion**

In conclusion, it makes sense that a black occlusion patch would result in lower confidence scores for predictions, as a whole section of the image is just completely black, completely blocking any details that would allow for higher confidence scores.

It also makes sense that a blur patch would result in funky results, as when something is blurred it can be hard to make out exacly what it is, as important details are smoothed out, but form and color still exist, resulting in predictions that do take some details into account but not enough to have a more refined and accurate guess.

---

# Creating and Experimenting with Image Filters

## AI Explanation

The AI's explanation of the **basic_filter.py** file's code did make sense to me, even more-so after implementing the logic for image classification and adding occlusion patches.

Just as was done previously for **base_classifier.py**, the image is resized, but in this case it adds the GuassianBlur filter using PIL.Image and ImageFilter. Just as I did previously in **base_classifier.py**, it uses plt.imshow, plt.axis('off'), and plt.savefig(*image_path*) to display and save the image. It also calls the function in a similar block of code beginning with *if __name__ == "__main__":* and passes in the image path.

After implementing the logic in **base_classifier.py**, the **basic_filter.py** code seems far more simple, basic, and, above all, easy to understand.

## Custom Filter (Posterize then Pixelate)
# Alternate Title: Minecraft Artwork

The custom filter that I ultimately ended up creating is a combination of two other custom filters: posterize and pixelate. 

The first custom filter I implemented was posterize, which gave the image a cool cartoon-adjacent/comic-book-adjacent aesthetic, but I felt that I could do more with it.

The second custom filter I implemented was pixelate, which pixelates the image using a pixel size of 2 to preserve details, rather than going with a chunkier pixelation, which obscures detail and makes the image unrecognizable.

I decided to experiment by combining these two filters, first by pixelating and then posterizing, and then by posterizing and then pixelating. 
Performing the pixelation before posterization made the image sort of awkward and not very aesthetically pleasing. 
However, working the other way around, and posterizing then pixelating, gave the image a much cooler aesthetic than either posterize or pixelate did alone, individually. 

In my opinion, this filter makes an image look like the artwork that you can hang on the walls in Minecraft. 
The colors are smooth and vibrant and the pixels give it a retro feel.
Ultimately, I am very pleased with the custom filter I ended up creating and implementing, with the help of the AI agent.

## Reflection

Working with the AI to explain and write Python code was enlightening and entertaining. I had fun collaborating with the AI to bring my ideas to life, and the AI's explanations of the code and detailed answers to clarifying questions I would ask were incredibly educational and enlightening and helped to train how I thought about the program and about coding in Python in general.

Using the AI, I was more quickly able to bring to life the custom filter I had imagined, and any tweaks or revisions I wished to make to the code and the filters along the way were much easier to make with an AI agent available to provide detailed context and instructions in exchange for code revisions and additions. In a way, this made working on the project fun and entertaining, as having the AI to assist alleviated much of the stress and headaches that come along with coding. This meant that I was able to let my creativity shine bright without being impeded by frustration and confusion, leading to a satisfying outcome and a lack of stress.

The AI does a great job of not only explaining the code, but explaining why it wrote the code one way instead of the other. It also was very responsive to my own suggestions or corrections when it would spit out redundant code for something that I had already created a helper function to handle. The AI also made it extremely easy to refactor my code or adjust the logic to achieve slightly different results in one way or another.

Working with the AI also proved to make debugging far easier, as I could paste error messages along with the lines of code they respond to, and the AI could detect what the issue was, whether it was an import I was missing, a syntax issue, or otherwise. This meant that I spent far less time trying to fix my code and more time trying to get creative with figuring out what I wanted my code to do. In a way, using the AI allowed me to keep looking ahead rather than looking back at the code or staying in the same spot debugging.

In conclusion, collaborating with the AI to write the code for this project was fun, educational, and alleviated stress and confusion. It proved to be a worthy tool for programming, and aided me in achieving a final result that I am very satisfied with.