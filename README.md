# SNN

Woke up Thanksgiving morning 2024 and thought this was a good idea...

## Main Functionality
At the center of the page, the user may upload an image from any anime (or any image in general). A backend neural network will try to guess what show it is from. The top 5 neural network predictions will be shown, along with the confidence of the guesses and associated MAL information if available. Users can experiement with testing where the model excels, and where it struggles.

## Games
### Filtering
In the upper right, users can optionally select what anime they've seen so that they only encounter these shows in the games.

### Title Guessing Game
The player is given a random frame from a random anime, and asked to guess what show it is from.

### Matching Game
The player is shown a grid of cards, behind each of which is an anime frame. There are 6 pairs in total, and the player must guess which cards are from the same show to make matches.

### Frame Guessing Game
The player is given an anime title, then quizzed on which of the six displayed frames is from that show.

## Neural Network Details
Exact network architecture/training details are left out of this repository for confidentiality purposes.

### Architecture
A wide range of architectures were tested, including existing models such as ResNet-18. While transfer learning from existing models was attempted, not much was used. As the number of anime the network trained on increased, the architecture grew as well.

### Training
The model was saved at each epoch, and trained intentionally until it overfit. Then the model saved at each epoch was evaluated an additional time to determine which model was "right before" it started overfitting to a considerable extent. This approach was taken since the model's need to generalize is relatively low (compared to other deep learning tasks). The model only needs to recognize images from similar anime, not anime its never seen before, or non-anime images. Future updates may include models that can identify anime character drawings or non-show images, which would involve models that can better generalize.

### Hardware
Current model: a mix of 48 cpus and 8 gpus over 2 days




# If you're interested...
Email me with additional fun ideas: howardluo2021@gmail.com
