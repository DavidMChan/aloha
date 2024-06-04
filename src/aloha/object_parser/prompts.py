SINGLE_OBJECT_SYSTEM_PROMPT = """You are an assistant that parses visually present objects from an image caption. Given an image caption, you list ALL the objects visually present in the image or photo described by the captions.

Strictly abide by the following rules:
- Include all attributes and adjectives that describe the object, if present
- Do not repeat objects
- Do not include objects that are mentioned but have no visual presence in the image, such as light, sound, or emotions
- If the caption is uncertain about an object, YOU MUST include '(possibly)' after the object
- If the caption thinks an object can be one of several things, include 'or' and all the possible objects
- Always give the singular form of the object, even if the caption uses the plural form
"""

MULTI_OBJECT_SYSTEM_PROMPT = """You are an assistant that parses visually present objects from a set of image captions. Given a set of image captions, you list ALL the objects visually present in the image or photo described by the captions.

Strictly abide by the following rules:
- Include all attributes and adjectives that describe the object, if present
- Do not repeat objects
- Do not include objects that are mentioned but have no visual presence in the image, such as light, sound, or emotions
- If the caption is uncertain about an object, YOU MUST include '(possibly)' after the object
- If the caption thinks an object can be one of several things, include 'or' and all the possible objects
- Always give the singular form of the object, even if the caption uses the plural form
"""

SINGLE_OBJECT_EXAMPLES = [
    (
        """Caption: Two women playing violin and banjo in front of a window.
Objects:""",
        """- woman
- violin
- banjo
- window""",
    ),
    (
        """Caption: A group of young boys sit at their desks in a classroom in the democratic republic of the congo.
Objects:""",
        """- young boy
- desk
- classroom""",
    ),
    (
        """Caption: A man in a black tank top or shirt is getting his hair braided.
Objects:""",
        """- man
- black tank top or shirt
- hair""",
    ),
    (
        """Caption: A happy kid is eating a cookie that looks like a christmas tree.
Objects:""",
        """- happy kid
- cookie""",
    ),
    (
        """Caption: A desk with two computer monitors, a laptop, a gray keyboard, printer or desktop, and possibly other items, such as papers, notebooks, and other clutter.
Objects:""",
        """- desk
- computer monitor
- laptop
- gray keyboard
- printer or desktop
- paper (possibly)
- notebook (possibly)""",
    ),
    (
        """Caption: Two women in white shirts playing a game of tennis on a court, with one of them wiping sweat from her face, with a pained facial expression, possibly due to frustration from losing a match, with people spectating.
Objects:""",
        """- woman
- white shirt
- tennis
- court
- face
- person""",
    ),
    (
        """Caption: A dimly lit kitchen with a sink, stove, cupboards, bottles, cups or jugs, and other items, white tiled walls, and wooden countertops, with light shining from a window.
Objects:""",
        """- dimly lit kitchen
- sink
- stove
- cupboard
- bottle
- cup or jug
- white tiled wall
- wooden countertop
- window""",
    ),
    (
        """Caption: This image shows two pink roses in a tulip-shaped vase on a wooden kitchen counter, next to a microwave and a toaster oven.
Objects:""",
        """- pink rose
- tulip-shaped vase
- wooden kitchen counter
- microwave
- toaster oven""",
    ),
    (
        """Caption: The image depicts a man wearing a checkered shirt, leaning forward to eat hotdogs on buns. He has a smirk on his face and is sitting on a styrofoam chair. The background of the image is black and white.
Objects:""",
        """- man
- checkered shirt
- hotdog
- bun
- face
- styrofoam chair""",
    ),
    (
        """Caption: A group of people between the ages of twenty and thirty-five are playing a nintendo wii game in a restaurant setting. They are wearing shorts and t-shirts and appear to be having a good time.
Objects:""",
        """- person
- nintendo wii game
- restaurant
- shorts
- t-shirt""",
    ),
]

MULTI_OBJECT_EXAMPLES = [
    (
        """Captions:
- A baseball player runs into home plate during a game.
- A man is running the bases in a baseball game.
- A group of players playing a baseball game.
- A man is running to home base in a baseball game.
- A man in a white shirt and gray pants walks toward a grassy area as kids in baseball uniforms and an umpire are near him.
Objects:""",
        """- baseball player
- home plate
- man
- base
- home base
- baseball game
- white shirt
- gray pants
- grassy area
- kid
- baseball uniform
- umpire""",
    ),
    (
        """Captions:
- Several people riding on a motorcycle with an umbrella open.
- Couples riding motor cycles carrying umbrellas and people sitting at tables.
- A group of people riding scooters while holding umbrellas.
- Some tables and umbrellas sitting next to a building.
- Pedestrians and motorcyclists near an open outdoor market.
Objects:""",
        """- person
- couple
- motorcycle
- umbrella
- table
- scooter
- building
- pedestrian
- motorcyclist
- open outdoor market
""",
    ),
    (
        """Captions:
- A person standing next to the water and a umbrella.
- The man rides the skateboard next to the water and the umbrella.
- A man standing on a pier overlooking the lake with an umbrella nearby.
- A man riding a skate board on railings near waterfront.
- A person is standing at the side of a big lake.
Objects:""",
        """- water
- umbrella
- man
- person
- skateboard
- pier
- lake
- big lake
- skateboard
- railing
- waterfront""",
    ),
]
