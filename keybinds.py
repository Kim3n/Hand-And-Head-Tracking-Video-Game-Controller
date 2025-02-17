# Key mappings and display text for each gesture
# Display text is optional, if left empty will show the key being pressed.
# Display text is more meant for Emulators so if you have 'W' set to 'D-pad up' you can show that on screen by doing left_hand_fist = ("w", "D-pad up")     


#SET YOUR KEYBINDINGS

# Left Hand Gesture Mappings

#OPEN AND CLOSED HAND
left_hand_fist = ("", "")                       # Left Hand open
left_hand_open = ("space", "")                # Left Hand fist 

#LEFT HAND SINGLE FINGER MAPPINGS
left_hand_thumb = ("", "")              # Thumb
left_hand_index = ("", "")                    # Index 
left_hand_middle = ("", "")                     # Middle
left_hand_ring = ("", "")                       # Ring 
left_hand_pinky = ("shift", "")                    # Pinky

#LEFT HAND TWO FINGER MAPPINGS
left_hand_thumb_index = ("", "")      # Thumb + Index Press
left_hand_thumb_middle = ("mouse click left", "")               # Thumb + Middle Press
left_hand_thumb_ring = ("", "")                 # Thumb + Ring Press
left_hand_thumb_pinky = ("", "")                # Thumb + Pinky Press
left_hand_index_middle = ("", "")               # Index + Middle Press
left_hand_index_ring= ("", "")                  # Index + Ring Press
left_hand_index_pinky = ("", "")                # Index + Pinky Press
left_hand_middle_ring = ("", "")                # Middle + Ring Press
left_hand_middle_pinky = ("", "")               # Middle + Pinky Press
left_hand_ring_pinky = ("", "")                 # Ring + Pinky Press

# LEFT HAND THREE FINGER MAPPINGS
left_hand_thumb_index_middle = ("", "")         # Thumb + Index + Middle Press
left_hand_thumb_index_ring = ("", "")           # Thumb + Index + Ring Press
left_hand_thumb_index_pinky = ("", "")          # Thumb + Index + Pinky Press
left_hand_thumb_middle_ring = ("", "")          # Thumb + Middle + Ring Press
left_hand_thumb_middle_pinky = ("", "")         # Thumb + Middle + Pinky Press
left_hand_thumb_ring_pinky = ("", "")           # Thumb + Ring + Pinky Press
left_hand_index_middle_ring = ("", "")          # Index + Middle + Ring Press
left_hand_index_middle_pinky = ("", "")         # Index + Middle + Pinky Press
left_hand_index_ring_pinky = ("", "")           # Index + Ring + Pinky Press
left_hand_middle_ring_pinky = ("", "")          # Middle + Ring + Pinky Press

# LEFT HAND FOUR FINGER MAPPINGS
left_hand_thumb_index_middle_ring = ("", "")    # Thumb + Index + Middle + Ring Press
left_hand_thumb_index_middle_pinky = ("", "")   # Thumb + Index + Middle + Pinky Press
left_hand_thumb_index_ring_pinky = ("", "")     # Thumb + Index + Ring + Pinky Press
left_hand_thumb_middle_ring_pinky = ("", "")    # Thumb + Middle + Ring + Pinky Press
left_hand_index_middle_ring_pinky = ("r", "")    # Index + Middle + Ring + Pinky Press


  



# Right Hand Gesture Mappings

# OPEN AND CLOSED HAND
right_hand_fist = ("", "")                           # Right Hand open
right_hand_open = ("e", "")                     # Right Hand fist

# RIGHT HAND SINGLE FINGER MAPPINGS
right_hand_thumb = ("", "")                # Thumb
right_hand_index = ("", "")                        # Index
right_hand_middle = ("", "")                         # Middle
right_hand_ring = ("", "")                           # Ring
right_hand_pinky = ("", "")                        # Pinky

# RIGHT HAND TWO FINGER MAPPINGS
right_hand_thumb_index = ("a", "")    # Thumb + Index Press
right_hand_thumb_middle = ("tab", "")              # Thumb + Middle Press
right_hand_thumb_ring = ("", "")                # Thumb + Ring Press
right_hand_thumb_pinky = ("d", "")               # Thumb + Pinky Press
right_hand_index_middle = ("s", "")              # Index + Middle Press
right_hand_index_ring = ("", "")                # Index + Ring Press
right_hand_index_pinky = ("w", "")               # Index + Pinky Press
right_hand_middle_ring = ("", "")               # Middle + Ring Press
right_hand_middle_pinky = ("", "")              # Middle + Pinky Press
right_hand_ring_pinky = ("", "")                # Ring + Pinky Press

# RIGHT HAND THREE FINGER MAPPINGS
right_hand_thumb_index_middle = ("", "")        # Thumb + Index + Middle Press
right_hand_thumb_index_ring = ("", "")          # Thumb + Index + Ring Press
right_hand_thumb_index_pinky = ("", "")         # Thumb + Index + Pinky Press
right_hand_thumb_middle_ring = ("", "")         # Thumb + Middle + Ring Press
right_hand_thumb_middle_pinky = ("", "")        # Thumb + Middle + Pinky Press
right_hand_thumb_ring_pinky = ("", "")          # Thumb + Ring + Pinky Press
right_hand_index_middle_ring = ("", "")         # Index + Middle + Ring Press
right_hand_index_middle_pinky = ("", "")        # Index + Middle + Pinky Press
right_hand_index_ring_pinky = ("", "")          # Index + Ring + Pinky Press
right_hand_middle_ring_pinky = ("", "")         # Middle + Ring + Pinky Press

# RIGHT HAND FOUR FINGER MAPPINGS
right_hand_thumb_index_middle_ring = ("", "")   # Thumb + Index + Middle + Ring Press
right_hand_thumb_index_middle_pinky = ("", "")  # Thumb + Index + Middle + Pinky Press
right_hand_thumb_index_ring_pinky = ("", "")    # Thumb + Index + Ring + Pinky Press
right_hand_thumb_middle_ring_pinky = ("", "")   # Thumb + Middle + Ring + Pinky Press
right_hand_index_middle_ring_pinky = ("", "")   # Index + Middle + Ring + Pinky Press



# Function to get all the left hand keypress and corresponding display text based on finger status
def get_left_hand_key(fingerStatus):
    # ALL ONE FINGER GESTURES (gesture 1)
    if fingerStatus == [1, 0, 0, 0, 0]:  # Thumb
        key, text = left_hand_thumb
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 0, 0, 0]:  # Index
        key, text = left_hand_index
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 1, 0, 0]:  # Middle
        key, text = left_hand_middle
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 0, 1, 0]:  # Ring
        key, text = left_hand_ring
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 0, 0, 1]:  # Pinky
        key, text = left_hand_pinky
        return (key, text if text else key)

    # ALL TWO FINGER GESTURES (gesture 2)
    elif fingerStatus == [1, 1, 0, 0, 0]:  # Thumb + Index
        key, text = left_hand_thumb_index
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 1, 0, 0]:  # Thumb + Middle
        key, text = left_hand_thumb_middle
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 0, 1, 0]:  # Thumb + Ring
        key, text = left_hand_thumb_ring
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 0, 0, 1]:  # Thumb + Pinky
        key, text = left_hand_thumb_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 1, 0, 0]:  # Index + Middle
        key, text = left_hand_index_middle
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 0, 1, 0]:  # Index + Ring
        key, text = left_hand_index_ring
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 0, 0, 1]:  # Index + Pinky
        key, text = left_hand_index_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 1, 1, 0]:  # Middle + Ring
        key, text = left_hand_middle_ring
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 1, 0, 1]:  # Middle + Pinky
        key, text = left_hand_middle_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 0, 1, 1]:  # Ring + Pinky
        key, text = left_hand_ring_pinky
        return (key, text if text else key)

    # ALL THREE FINGER GESTURES (gesture 3)
    elif fingerStatus == [1, 1, 1, 0, 0]:  # Thumb + Index + Middle
        key, text = left_hand_thumb_index_middle
        return (key, text if text else key)
    elif fingerStatus == [1, 1, 0, 1, 0]:  # Thumb + Index + Ring
        key, text = left_hand_thumb_index_ring
        return (key, text if text else key)
    elif fingerStatus == [1, 1, 0, 0, 1]:  # Thumb + Index + Pinky
        key, text = left_hand_thumb_index_pinky
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 1, 1, 0]:  # Thumb + Middle + Ring
        key, text = left_hand_thumb_middle_ring
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 1, 0, 1]:  # Thumb + Middle + Pinky
        key, text = left_hand_thumb_middle_pinky
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 0, 1, 1]:  # Thumb + Ring + Pinky
        key, text = left_hand_thumb_ring_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 1, 1, 0]:  # Index + Middle + Ring
        key, text = left_hand_index_middle_ring
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 1, 0, 1]:  # Index + Middle + Pinky
        key, text = left_hand_index_middle_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 0, 1, 1]:  # Index + Ring + Pinky
        key, text = left_hand_index_ring_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 1, 1, 1]:  # Middle + Ring + Pinky
        key, text = left_hand_middle_ring_pinky
        return (key, text if text else key)

    # ALL FOUR FINGER GESTURES (gesture 4)
    elif fingerStatus == [1, 1, 1, 1, 0]:  # Thumb + Index + Middle + Ring
        key, text = left_hand_thumb_index_middle_ring
        return (key, text if text else key)
    elif fingerStatus == [1, 1, 1, 0, 1]:  # Thumb + Index + Middle + Pinky
        key, text = left_hand_thumb_index_middle_pinky
        return (key, text if text else key)
    elif fingerStatus == [1, 1, 0, 1, 1]:  # Thumb + Index + Ring + Pinky
        key, text = left_hand_thumb_index_ring_pinky
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 1, 1, 1]:  # Thumb + Middle + Ring + Pinky
        key, text = left_hand_thumb_middle_ring_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 1, 1, 1]:  # Index + Middle + Ring + Pinky
        key, text = left_hand_index_middle_ring_pinky
        return (key, text if text else key)

    # ALL FIVE FINGERS GESTURE (gesture 5)
    elif fingerStatus == [1, 1, 1, 1, 1]:  # Thumb + Index + Middle + Ring + Pinky
        key, text = left_hand_open
        return (key, text if text else key)

    # If none of the conditions match, return empty values
    else:
        return ("", "")

# Function to get all the right hand keypress and corresponding display text based on finger status
def get_right_hand_key(fingerStatus):
    # ALL ONE FINGER GESTURES (gesture 1)
    if fingerStatus == [1, 0, 0, 0, 0]:  # Thumb
        key, text = right_hand_thumb
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 0, 0, 0]:  # Index
        key, text = right_hand_index
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 1, 0, 0]:  # Middle
        key, text = right_hand_middle
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 0, 1, 0]:  # Ring
        key, text = right_hand_ring
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 0, 0, 1]:  # Pinky
        key, text = right_hand_pinky
        return (key, text if text else key)

    # ALL TWO FINGER GESTURES (gesture 2)
    elif fingerStatus == [1, 1, 0, 0, 0]:  # Thumb + Index
        key, text = right_hand_thumb_index
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 1, 0, 0]:  # Thumb + Middle
        key, text = right_hand_thumb_middle
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 0, 1, 0]:  # Thumb + Ring
        key, text = right_hand_thumb_ring
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 0, 0, 1]:  # Thumb + Pinky
        key, text = right_hand_thumb_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 1, 0, 0]:  # Index + Middle
        key, text = right_hand_index_middle
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 0, 1, 0]:  # Index + Ring
        key, text = right_hand_index_ring
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 0, 0, 1]:  # Index + Pinky
        key, text = right_hand_index_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 1, 1, 0]:  # Middle + Ring
        key, text = right_hand_middle_ring
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 1, 0, 1]:  # Middle + Pinky
        key, text = right_hand_middle_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 0, 1, 1]:  # Ring + Pinky
        key, text = right_hand_ring_pinky
        return (key, text if text else key)

    # ALL THREE FINGER GESTURES (gesture 3)
    elif fingerStatus == [1, 1, 1, 0, 0]:  # Thumb + Index + Middle
        key, text = right_hand_thumb_index_middle
        return (key, text if text else key)
    elif fingerStatus == [1, 1, 0, 1, 0]:  # Thumb + Index + Ring
        key, text = right_hand_thumb_index_ring
        return (key, text if text else key)
    elif fingerStatus == [1, 1, 0, 0, 1]:  # Thumb + Index + Pinky
        key, text = right_hand_thumb_index_pinky
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 1, 1, 0]:  # Thumb + Middle + Ring
        key, text = right_hand_thumb_middle_ring
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 1, 0, 1]:  # Thumb + Middle + Pinky
        key, text = right_hand_thumb_middle_pinky
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 0, 1, 1]:  # Thumb + Ring + Pinky
        key, text = right_hand_thumb_ring_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 1, 1, 0]:  # Index + Middle + Ring
        key, text = right_hand_index_middle_ring
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 1, 0, 1]:  # Index + Middle + Pinky
        key, text = right_hand_index_middle_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 0, 1, 1]:  # Index + Ring + Pinky
        key, text = right_hand_index_ring_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 0, 1, 1, 1]:  # Middle + Ring + Pinky
        key, text = right_hand_middle_ring_pinky
        return (key, text if text else key)

    # ALL FOUR FINGER GESTURES (gesture 4)
    elif fingerStatus == [1, 1, 1, 1, 0]:  # Thumb + Index + Middle + Ring
        key, text = right_hand_thumb_index_middle_ring
        return (key, text if text else key)
    elif fingerStatus == [1, 1, 1, 0, 1]:  # Thumb + Index + Middle + Pinky
        key, text = right_hand_thumb_index_middle_pinky
        return (key, text if text else key)
    elif fingerStatus == [1, 1, 0, 1, 1]:  # Thumb + Index + Ring + Pinky
        key, text = right_hand_thumb_index_ring_pinky
        return (key, text if text else key)
    elif fingerStatus == [1, 0, 1, 1, 1]:  # Thumb + Middle + Ring + Pinky
        key, text = right_hand_thumb_middle_ring_pinky
        return (key, text if text else key)
    elif fingerStatus == [0, 1, 1, 1, 1]:  # Index + Middle + Ring + Pinky
        key, text = right_hand_index_middle_ring_pinky
        return (key, text if text else key)

    # ALL FIVE FINGERS GESTURE (gesture 5)
    elif fingerStatus == [1, 1, 1, 1, 1]:  # Thumb + Index + Middle + Ring + Pinky
        key, text = right_hand_open
        return (key, text if text else key)

    # If none of the conditions match, return empty values
    else:
        return ("", "")
