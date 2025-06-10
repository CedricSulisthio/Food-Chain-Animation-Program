# Program 1

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Constants
TIME_STEP = 0.1
RABBIT_LIFETIME = 15  # Rabbit disappears after 15 seconds
FOX_LIFETIME = 15  # Fox disappears after 15 seconds
CARROT_LIFETIME_EXTENSION = 20  # Rabbit's lifetime extends by 20 seconds
FOX_LIFETIME_EXTENSION = 20  # Fox's lifetime extends by 20 seconds
SENSING_RANGE = 1000.0  # Large sensing range to detect all entities
AVOIDANCE_RANGE = 1.5  # Distance at which rabbits will try to avoid foxes and other rabbits
FIELD_LIMIT = 15  # Expanded field boundary
IMAGE_SIZE = 0.15  # Default image size
GROWTH_RATE = 0.01  # Growth increment for eating
TOUCHING_RANGE = 1.0  # Range to consider interaction (e.g., carrot eaten or rabbit caught)

# Rabbit, Fox, and Carrot classes
class Rabbit:
    def __init__(self, alpha=1.0):
        self.position = np.random.uniform(-FIELD_LIMIT, FIELD_LIMIT, 2)  # Random initial position
        self.velocity = np.random.uniform(-1, 1, 2)  # Random initial velocity
        self.acceleration = np.array([0.0, 0.0])  # Initial acceleration
        self.alive_time = 0  # Time since the rabbit was spawned
        self.alpha = alpha  # Sensitivity level
        self.image_size = IMAGE_SIZE  # Initial image size

    def is_dead(self):
        return self.alive_time >= RABBIT_LIFETIME

    def grow(self):
        self.image_size += GROWTH_RATE  # Gradual growth

class Fox:
    def __init__(self, alpha=1.0):
        self.position = np.random.uniform(-FIELD_LIMIT, FIELD_LIMIT, 2)  # Random initial position
        self.velocity = np.random.uniform(-1, 1, 2)  # Random initial velocity
        self.acceleration = np.array([0.0, 0.0])  # Initial acceleration
        self.alive_time = 0  # Time since the fox was spawned
        self.alpha = alpha  # Sensitivity level
        self.image_size = IMAGE_SIZE  # Initial image size

    def is_dead(self):
        return self.alive_time >= FOX_LIFETIME

    def grow(self):
        self.image_size += GROWTH_RATE  # Gradual growth

class Carrot:
    def __init__(self):
        while True:
            self.position = np.random.uniform(-FIELD_LIMIT, FIELD_LIMIT, 2)
            # Ensure no overlap with existing carrots
            if all(np.linalg.norm(self.position - c.position) > 1.5 for c in carrots):
                break

# Helper functions for updating entities and interactions

def update_rabbit(rabbit, carrots, foxes, other_rabbits):
    rabbit.acceleration = np.array([0.0, 0.0])  # Reset acceleration

    # Rabbit chases carrots
    if carrots:
        distances = [np.linalg.norm(carrot.position - rabbit.position) for carrot in carrots]
        nearest_index = np.argmin(distances)
        target_carrot = carrots[nearest_index]
        diff = target_carrot.position - rabbit.position
        distance = np.linalg.norm(diff)
        if distance < SENSING_RANGE:
            rabbit.acceleration += rabbit.alpha * diff / (distance + 1e-5)

    # Rabbit avoids foxes
    for fox in foxes:
        diff = rabbit.position - fox.position
        distance = np.linalg.norm(diff)
        if distance < SENSING_RANGE:
            rabbit.acceleration += rabbit.alpha * diff / (distance + 1e-5)

    # Rabbit avoids other rabbits
    for other in other_rabbits:
        if other is not rabbit:
            diff = rabbit.position - other.position
            distance = np.linalg.norm(diff)
            if distance < AVOIDANCE_RANGE:
                rabbit.acceleration += rabbit.alpha * diff / (distance + 1e-5)

    # Random movement if no carrots or foxes detected
    if not carrots:
        rabbit.acceleration += np.random.uniform(-1, 1, 2)

    # Update velocity and position
    rabbit.velocity += rabbit.acceleration * TIME_STEP
    rabbit.position += rabbit.velocity * TIME_STEP
    rabbit.position = np.clip(rabbit.position, -FIELD_LIMIT, FIELD_LIMIT)  # Stay within bounds
    rabbit.alive_time += TIME_STEP

    return rabbit

def update_fox(fox, rabbits):
    fox.acceleration = np.array([0.0, 0.0])  # Reset acceleration

    # Fox chases rabbits
    if rabbits:
        distances = [np.linalg.norm(rabbit.position - fox.position) for rabbit in rabbits]
        nearest_index = np.argmin(distances)
        target_rabbit = rabbits[nearest_index]
        diff = target_rabbit.position - fox.position
        distance = np.linalg.norm(diff)
        if distance < SENSING_RANGE:
            fox.acceleration += fox.alpha * diff / (distance + 1e-5)

    # Random movement if no rabbits detected
    if not rabbits:
        fox.acceleration += np.random.uniform(-1, 1, 2)

    # Update velocity and position
    fox.velocity += fox.acceleration * TIME_STEP
    fox.position += fox.velocity * TIME_STEP
    fox.position = np.clip(fox.position, -FIELD_LIMIT, FIELD_LIMIT)  # Stay within bounds
    fox.alive_time += TIME_STEP

    return fox

def update_carrot(carrot, rabbits):
    # Carrots don't move, but we can check for rabbit interactions
    for rabbit in rabbits:
        if np.linalg.norm(carrot.position - rabbit.position) < TOUCHING_RANGE:
            # Rabbit eats carrot
            return None  # Remove carrot

    return carrot

def handle_rabbit_fox_interaction(rabbits, foxes):
    rabbits_to_remove = []
    for i in range(len(rabbits)):
        rabbit = rabbits[i]
        for j in range(len(foxes)):
            fox = foxes[j]
            if np.linalg.norm(fox.position - rabbit.position) < TOUCHING_RANGE:
                # Fox catches rabbit
                rabbits_to_remove.append(i)
                fox.alive_time -= FOX_LIFETIME_EXTENSION  # Extend fox's lifetime
                fox.grow()  # Fox grows after eating rabbit
                break  # No need to check other foxes for this rabbit

    return rabbits_to_remove

# Main update function that integrates all the updates and rendering
def update(frame):
    updated_artists = []

    # 1. Update rabbits
    for i in range(len(rabbits)):
        rabbits[i] = update_rabbit(rabbits[i], carrots, foxes, rabbits)

    # 2. Update foxes
    for i in range(len(foxes)):
        foxes[i] = update_fox(foxes[i], rabbits)

    # 3. Handle rabbit-fox interactions
    rabbits_to_remove = handle_rabbit_fox_interaction(rabbits, foxes)

    # 4. Remove caught rabbits
    for index in reversed(rabbits_to_remove):
        rabbit_annotations[index].remove()
        del rabbit_annotations[index]
        del rabbits[index]

    # 5. Update carrots
    carrots_to_remove = []
    for i in range(len(carrots)):
        carrots[i] = update_carrot(carrots[i], rabbits)
        if carrots[i] is None:  # Carrot was eaten
            carrots_to_remove.append(i)

    # 6. Remove eaten carrots
    for index in reversed(carrots_to_remove):
        carrot_annotations[index].remove()
        del carrot_annotations[index]
        del carrots[index]

    # 7. Render updated entities
    updated_artists = []
    for i in range(len(rabbits)):
        rabbit_annotations[i].remove()
        rabbit_annotations[i] = add_image(ax, rabbit_image, rabbits[i].position, rabbits[i].image_size)
        updated_artists.append(rabbit_annotations[i])

    for i in range(len(foxes)):
        fox_annotations[i].remove()
        fox_annotations[i] = add_image(ax, fox_image, foxes[i].position, foxes[i].image_size)
        updated_artists.append(fox_annotations[i])

    for i in range(len(carrots)):
        carrot_annotations[i].remove()
        carrot_annotations[i] = add_image(ax, carrot_image, carrots[i].position, IMAGE_SIZE)
        updated_artists.append(carrot_annotations[i])

    return updated_artists

# Helper function to add images
def add_image(ax, image, position, size):
    imagebox = OffsetImage(image, zoom=size)
    ab = AnnotationBbox(imagebox, position, frameon=False)
    ax.add_artist(ab)
    return ab

# Initialize entities
rabbits = []
foxes = []
carrots = []

# Create the plot
fig, ax = plt.subplots()
ax.set_xlim(-FIELD_LIMIT, FIELD_LIMIT)
ax.set_ylim(-FIELD_LIMIT, FIELD_LIMIT)
ax.set_title("Food Chain Simulation")

# Load images
rabbit_image = plt.imread("C:/Users/Cedric Sulisthio/Documents/IBEProjectsandTutor/Difference Equations/Rabbit.jpg")
fox_image = plt.imread("C:/Users/Cedric Sulisthio/Documents/IBEProjectsandTutor/Difference Equations/Fox.png")
carrot_image = plt.imread("C:/Users/Cedric Sulisthio/Documents/IBEProjectsandTutor/Difference Equations/Carrot.jpg")

rabbit_annotations = []
fox_annotations = []
carrot_annotations = []

# Functions to spawn entities
def spawn_rabbit(event):
    new_rabbit = Rabbit(alpha=np.random.uniform(0.5, 1.5))
    rabbits.append(new_rabbit)
    rabbit_annotations.append(add_image(ax, rabbit_image, new_rabbit.position, new_rabbit.image_size))

def spawn_fox(event):
    new_fox = Fox(alpha=np.random.uniform(0.5, 1.5))
    foxes.append(new_fox)
    fox_annotations.append(add_image(ax, fox_image, new_fox.position, new_fox.image_size))

def spawn_carrot(event):
    new_carrot = Carrot()
    carrots.append(new_carrot)
    carrot_annotations.append(add_image(ax, carrot_image, new_carrot.position, IMAGE_SIZE))

# Set up the animation
ani = animation.FuncAnimation(fig, update, interval=100, blit=True)

# Add buttons for spawning entities
ax_rabbit_button = plt.axes([0.1, 0.01, 0.2, 0.05])
rabbit_button = Button(ax_rabbit_button, "Spawn Rabbit")
rabbit_button.on_clicked(spawn_rabbit)

ax_fox_button = plt.axes([0.4, 0.01, 0.2, 0.05])
fox_button = Button(ax_fox_button, "Spawn Fox")
fox_button.on_clicked(spawn_fox)

ax_carrot_button = plt.axes([0.7, 0.01, 0.2, 0.05])
carrot_button = Button(ax_carrot_button, "Spawn Carrot")
carrot_button.on_clicked(spawn_carrot)

# Show the plot
plt.show()