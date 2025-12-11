from collections import deque, Counter

class StateStabilizer:
    def __init__(self, buffer_len=15, threshold=0.6):
        self.buffer = deque(maxlen=buffer_len)
        self.threshold = threshold # Require 60% agreement

    def update(self, current_frame_fen):
        """
        Pushes current frame's FEN (board string) to buffer.
        Returns the stable FEN if consensus is met, else None.
        """
        self.buffer.append(current_frame_fen)
        
        if len(self.buffer) < self.buffer.maxlen:
            return None # Wait for buffer to fill

        # Find most common board state
        most_common, count = Counter(self.buffer).most_common(1)[0]
        
        if count / len(self.buffer) >= self.threshold:
            return most_common
        return None