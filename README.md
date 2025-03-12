# Hand And Head Tracking Video Game Controller

This project uses Python and MediaPipe to turn hand and head tracking into video game inputs.

## How It Works

### Hand Tracking

The program tracks your hands and detects which fingers are up. Each hand position is represented as a five-digit binary list, where `1` means a finger is up and `0` means it's down.

#### Examples:

- `[0,0,0,0,0]` – No fingers up
- `[0,1,1,0,0]` – Index and middle fingers up

You can bind different finger combinations to specific keys. For example, `[0,1,1,0,0]` could be mapped to the **"S" key**.

### Head Tracking

The program tracks three key points on your face:

- **Nose tip**
- **Outer edge of left eye**
- **Outer edge of right eye**

By measuring the distance between the nose and these points, the program determines which direction you're looking:

- **Straight ahead** – No movement
- **Left / Right** – Moves the mouse horizontally in that direction
- **Up / Down** – Moves the mouse vertically in that direction

## Example Video

<video controls src="media/example.mp4" title="Example video"></video>
