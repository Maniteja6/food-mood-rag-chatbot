"""
generate_dataset.py
Generates a realistic 50,000-row food dataset for the MoodBite RAG chatbot.
"""

import random
import csv
import itertools
from pathlib import Path

random.seed(42)

# ─────────────────────────────────────────────────────────────
# RAW BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

CUISINES = [
    "Italian", "Japanese", "Indian", "Mexican", "Chinese",
    "Thai", "Mediterranean", "American", "French", "Korean",
    "Middle Eastern", "Spanish", "Greek", "Vietnamese", "Ethiopian",
    "Peruvian", "Turkish", "Moroccan", "Lebanese", "Brazilian",
    "British", "German", "Indonesian", "Filipino", "Caribbean",
    "Russian", "Polish", "Nigerian", "Argentinian", "Swedish",
]

MOODS_LIST = [
    "happy", "sad", "stressed", "tired", "romantic",
    "excited", "cozy", "adventurous", "anxious", "bored",
    "nostalgic", "celebratory", "lonely", "energetic", "sluggish",
    "focused", "heartbroken", "proud", "nervous", "content",
]

DIETARY_TAGS_POOL = [
    "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free",
    "Nut-Free", "Halal", "Kosher", "Keto", "Paleo",
    "Low-Carb", "High-Protein", "Low-Sodium", "Raw",
    "Whole30", "Pescatarian",
]

COOKING_METHODS = [
    "grilled", "baked", "fried", "steamed", "braised",
    "roasted", "poached", "sautéed", "slow-cooked", "smoked",
    "stir-fried", "deep-fried", "pressure-cooked", "raw", "cured",
    "fermented", "boiled", "pan-seared", "wok-tossed", "charcoal-grilled",
]

FLAVOUR_PROFILES = [
    "umami-rich", "tangy", "spicy", "sweet and sour", "smoky",
    "herbaceous", "creamy", "nutty", "citrusy", "earthy",
    "savoury", "bitter", "fragrant", "bold", "delicate",
    "buttery", "peppery", "garlicky", "sweet", "briny",
]

TEXTURES = [
    "crispy", "tender", "silky", "chunky", "velvety",
    "crunchy", "fluffy", "chewy", "melt-in-your-mouth", "hearty",
    "light", "dense", "flaky", "sticky", "airy",
]

OCCASIONS = [
    "weeknight dinner", "date night", "family gathering",
    "solo lunch", "office party", "meal prep", "Sunday brunch",
    "late-night snack", "post-workout meal", "hangover cure",
    "celebration feast", "comfort meal", "quick bite", "picnic",
    "holiday feast", "beach day", "rainy day", "camping trip",
    "potluck", "romantic dinner",
]

# Cuisine → base dishes (we'll generate many variations from these)
CUISINE_DISHES = {
    "Italian": [
        ("Spaghetti Carbonara", ["pasta","egg","pancetta","pecorino"], 25),
        ("Margherita Pizza", ["pizza dough","tomato","mozzarella","basil"], 30),
        ("Risotto ai Funghi", ["arborio rice","porcini","parmesan","white wine"], 40),
        ("Osso Buco", ["veal shank","gremolata","saffron risotto"], 90),
        ("Penne all'Arrabbiata", ["penne","tomato","chilli","garlic"], 20),
        ("Lasagne Bolognese", ["lasagne sheets","beef","béchamel","parmesan"], 75),
        ("Tiramisu", ["mascarpone","espresso","ladyfingers","cocoa"], 20),
        ("Cacio e Pepe", ["tonnarelli","pecorino","black pepper"], 15),
        ("Saltimbocca", ["veal","prosciutto","sage","white wine"], 20),
        ("Focaccia Genovese", ["bread dough","olive oil","rosemary","sea salt"], 60),
        ("Panzanella", ["stale bread","tomatoes","cucumber","basil"], 15),
        ("Ribollita", ["bread","cannellini beans","kale","vegetables"], 60),
        ("Gnocchi al Pomodoro", ["potato gnocchi","tomato sauce","basil"], 30),
        ("Arancini", ["risotto","mozzarella","breadcrumbs"], 40),
        ("Bruschetta al Pomodoro", ["sourdough","tomatoes","garlic","basil"], 10),
    ],
    "Japanese": [
        ("Ramen", ["noodles","broth","chashu pork","soft-boiled egg","nori"], 90),
        ("Sushi Platter", ["sushi rice","assorted fish","nori","wasabi"], 45),
        ("Chicken Katsu Curry", ["chicken","breadcrumbs","curry sauce","rice"], 35),
        ("Miso Soup", ["tofu","wakame","miso","dashi"], 10),
        ("Yakitori", ["chicken","tare sauce","scallions"], 25),
        ("Tonkatsu", ["pork loin","panko","cabbage","tonkatsu sauce"], 30),
        ("Tempura Udon", ["udon noodles","prawn tempura","dashi broth"], 30),
        ("Gyoza", ["pork mince","cabbage","ginger","dumpling wrappers"], 30),
        ("Okonomiyaki", ["cabbage","batter","pork","bonito flakes","mayo"], 25),
        ("Takoyaki", ["octopus","batter","bonito flakes","okonomiyaki sauce"], 30),
        ("Chirashi Don", ["sushi rice","assorted sashimi","pickles"], 20),
        ("Shabu Shabu", ["thin-sliced beef","vegetables","ponzu","sesame broth"], 30),
        ("Onigiri", ["sushi rice","umeboshi","nori","salmon"], 20),
        ("Chawanmushi", ["egg custard","dashi","shrimp","mitsuba"], 30),
        ("Matcha Ice Cream", ["matcha","cream","sugar","egg yolks"], 240),
    ],
    "Indian": [
        ("Butter Chicken", ["chicken","tomato","cream","spices"], 45),
        ("Palak Paneer", ["spinach","paneer","spices","cream"], 35),
        ("Biryani", ["basmati rice","chicken","saffron","fried onions"], 90),
        ("Dal Makhani", ["black lentils","butter","cream","tomatoes"], 120),
        ("Chana Masala", ["chickpeas","tomatoes","onions","spices"], 45),
        ("Aloo Gobi", ["potato","cauliflower","turmeric","cumin"], 30),
        ("Samosa", ["potato","peas","pastry","cumin"], 45),
        ("Tandoori Chicken", ["chicken","yoghurt","tandoori spices"], 240),
        ("Lamb Rogan Josh", ["lamb","Kashmiri chillies","yoghurt","whole spices"], 75),
        ("Pani Puri", ["puri","potato","chickpeas","tamarind water"], 30),
        ("Masala Dosa", ["rice batter","potato filling","chutney"], 40),
        ("Chole Bhature", ["chickpeas","fried bread","pickles"], 45),
        ("Gulab Jamun", ["milk powder","sugar syrup","cardamom"], 30),
        ("Raita", ["yoghurt","cucumber","cumin","mint"], 5),
        ("Kheer", ["rice","milk","cardamom","saffron"], 45),
    ],
    "Mexican": [
        ("Tacos al Pastor", ["pork","pineapple","corn tortillas","salsa verde"], 30),
        ("Beef Burrito", ["flour tortilla","beef","rice","beans","guacamole"], 25),
        ("Chicken Enchiladas", ["chicken","enchilada sauce","cheese","tortillas"], 45),
        ("Guacamole", ["avocado","lime","cilantro","jalapeño","onion"], 10),
        ("Quesadilla", ["flour tortilla","cheese","chicken","peppers"], 15),
        ("Pozole", ["hominy","pork","dried chillies","oregano"], 120),
        ("Mole Poblano", ["chicken","mole sauce","chocolate","dried chillies"], 120),
        ("Chilaquiles", ["tortilla chips","salsa","eggs","queso fresco"], 20),
        ("Tamales", ["masa","pork","dried chillies","corn husks"], 90),
        ("Elote", ["corn","mayo","cheese","chilli powder","lime"], 15),
        ("Huevos Rancheros", ["eggs","ranchero sauce","tortillas","black beans"], 20),
        ("Sopa de Lima", ["chicken","lime","tortilla strips","avocado"], 35),
        ("Ceviche Verde", ["fish","tomatillo","cucumber","avocado","lime"], 20),
        ("Churros", ["fried dough","cinnamon sugar","chocolate sauce"], 20),
        ("Horchata", ["rice","almonds","cinnamon","sugar"], 15),
    ],
    "Chinese": [
        ("Kung Pao Chicken", ["chicken","peanuts","dried chillies","sichuan pepper"], 25),
        ("Dim Sum Platter", ["har gow","siu mai","char siu bao","egg tart"], 45),
        ("Mapo Tofu", ["silken tofu","pork mince","doubanjiang","sichuan pepper"], 20),
        ("Peking Duck", ["whole duck","hoisin","pancakes","cucumber"], 120),
        ("Char Siu Pork", ["pork belly","hoisin","honey","five spice"], 90),
        ("Hot and Sour Soup", ["tofu","wood-ear mushrooms","bamboo shoots","vinegar"], 20),
        ("Spring Rolls", ["vegetables","glass noodles","wrappers","dipping sauce"], 30),
        ("Beef Chow Fun", ["flat rice noodles","beef","bean sprouts","soy sauce"], 20),
        ("Sweet and Sour Pork", ["pork","pineapple","peppers","sweet sour sauce"], 30),
        ("Dan Dan Noodles", ["noodles","pork mince","sesame paste","chilli oil"], 20),
        ("Egg Fried Rice", ["jasmine rice","eggs","spring onions","soy sauce"], 15),
        ("Wonton Soup", ["wonton dumplings","pork","prawn","broth"], 30),
        ("Lion's Head Meatballs", ["pork mince","cabbage","broth","ginger"], 45),
        ("Mango Pudding", ["mango","gelatin","cream","condensed milk"], 180),
        ("Scallion Pancakes", ["flour","scallions","sesame oil"], 30),
    ],
    "Thai": [
        ("Pad Thai", ["rice noodles","tofu","egg","tamarind","peanuts"], 20),
        ("Green Curry", ["coconut milk","green curry paste","chicken","eggplant"], 25),
        ("Tom Yum Soup", ["lemongrass","galangal","mushrooms","shrimp","lime"], 25),
        ("Som Tam", ["green papaya","chilli","lime","fish sauce","peanuts"], 10),
        ("Massaman Curry", ["beef","potato","coconut milk","peanuts","cardamom"], 90),
        ("Pad Kra Pao", ["minced pork","Thai basil","oyster sauce","chilli"], 15),
        ("Khao Man Gai", ["poached chicken","jasmine rice","ginger broth"], 45),
        ("Mango Sticky Rice", ["glutinous rice","ripe mango","coconut cream"], 30),
        ("Tom Kha Gai", ["chicken","coconut milk","galangal","mushrooms"], 25),
        ("Larb", ["minced meat","mint","fish sauce","lime","toasted rice"], 20),
        ("Pad See Ew", ["wide rice noodles","egg","Chinese broccoli","dark soy"], 20),
        ("Panang Curry", ["beef","Panang curry paste","coconut cream","kaffir lime"], 30),
        ("Crying Tiger", ["grilled beef","tamarind dipping sauce","nam jim"], 30),
        ("Kanom Krok", ["coconut milk","rice flour","spring onion"], 20),
        ("Boat Noodles", ["thin rice noodles","pork blood broth","herbs"], 20),
    ],
    "Mediterranean": [
        ("Greek Salad", ["tomatoes","cucumber","olives","feta","red onion"], 10),
        ("Hummus", ["chickpeas","tahini","lemon","garlic","olive oil"], 10),
        ("Falafel Wrap", ["falafel","pita","tahini","salad","pickles"], 30),
        ("Shakshuka", ["eggs","tomatoes","peppers","cumin","feta"], 25),
        ("Mezze Platter", ["hummus","baba ganoush","tabbouleh","pita","olives"], 20),
        ("Moussaka", ["aubergine","lamb mince","béchamel","tomatoes"], 75),
        ("Spanakopita", ["spinach","feta","filo pastry","eggs"], 60),
        ("Lamb Souvlaki", ["marinated lamb","pita","tzatziki","salad"], 30),
        ("Baba Ganoush", ["aubergine","tahini","lemon","garlic"], 30),
        ("Fattoush", ["toasted pita","mixed greens","radish","sumac dressing"], 15),
        ("Stuffed Vine Leaves", ["vine leaves","rice","pine nuts","lemon"], 60),
        ("Kleftiko", ["slow-roasted lamb","garlic","lemon","oregano"], 180),
        ("Pita Bread", ["flour","yeast","olive oil","salt"], 90),
        ("Baklava", ["filo pastry","walnuts","honey syrup","cinnamon"], 60),
        ("Labneh", ["strained yoghurt","olive oil","za'atar","herbs"], 480),
    ],
    "American": [
        ("Smash Burger", ["beef patty","american cheese","pickles","special sauce"], 15),
        ("BBQ Brisket", ["beef brisket","dry rub","BBQ sauce","coleslaw"], 480),
        ("Mac and Cheese", ["macaroni","cheddar","gruyere","breadcrumb topping"], 45),
        ("Buffalo Wings", ["chicken wings","buffalo sauce","blue cheese dip"], 45),
        ("New England Clam Chowder", ["clams","potato","cream","bacon"], 45),
        ("Lobster Roll", ["lobster","mayo","celery","buttered roll"], 20),
        ("Philly Cheesesteak", ["ribeye","provolone","onions","hoagie roll"], 20),
        ("Fried Chicken", ["chicken","buttermilk","seasoned flour","hot sauce"], 60),
        ("Pancakes with Maple Syrup", ["buttermilk","eggs","maple syrup","butter"], 20),
        ("Eggs Benedict", ["english muffin","poached eggs","hollandaise","bacon"], 25),
        ("Pulled Pork Sandwich", ["slow-cooked pork","BBQ sauce","brioche","pickles"], 480),
        ("New York Cheesecake", ["cream cheese","sour cream","graham cracker crust"], 90),
        ("Chicken and Waffles", ["fried chicken","waffles","maple syrup","hot sauce"], 45),
        ("Cobb Salad", ["chicken","bacon","avocado","blue cheese","hard-boiled egg"], 20),
        ("Key Lime Pie", ["lime juice","condensed milk","graham cracker crust","cream"], 60),
    ],
    "French": [
        ("Coq au Vin", ["chicken","red wine","mushrooms","lardons","pearl onions"], 90),
        ("French Onion Soup", ["onions","beef broth","gruyere","crouton"], 75),
        ("Croque Monsieur", ["ham","gruyere","béchamel","sourdough"], 15),
        ("Ratatouille", ["aubergine","courgette","tomatoes","peppers","herbes de Provence"], 60),
        ("Beef Bourguignon", ["beef","red wine","carrots","mushrooms","pearl onions"], 180),
        ("Bouillabaisse", ["mixed fish","saffron broth","rouille","baguette"], 60),
        ("Quiche Lorraine", ["pastry","eggs","cream","bacon","gruyere"], 60),
        ("Duck Confit", ["duck legs","garlic","thyme","duck fat"], 240),
        ("Crème Brûlée", ["cream","egg yolks","vanilla","caramel crust"], 60),
        ("Croissant", ["butter","flour","yeast","laminated dough"], 180),
        ("Soupe au Pistou", ["vegetables","basil","garlic","parmesan"], 45),
        ("Tarte Tatin", ["apples","caramel","puff pastry","butter"], 60),
        ("Steak Frites", ["ribeye","hand-cut fries","béarnaise","watercress"], 25),
        ("Salade Niçoise", ["tuna","green beans","olives","hard-boiled egg","anchovy"], 20),
        ("Profiteroles", ["choux pastry","vanilla cream","chocolate sauce"], 60),
    ],
    "Korean": [
        ("Bibimbap", ["rice","mixed vegetables","gochujang","fried egg","sesame oil"], 25),
        ("Korean Fried Chicken", ["chicken","gochujang glaze","pickled radish"], 45),
        ("Kimchi Jjigae", ["kimchi","pork belly","tofu","gochugaru"], 30),
        ("Japchae", ["glass noodles","spinach","mushrooms","carrots","sesame"], 30),
        ("Sundubu Jjigae", ["silken tofu","clams","gochugaru","egg"], 20),
        ("Galbi", ["beef short ribs","soy","pear","sesame","garlic"], 180),
        ("Samgyeopsal", ["grilled pork belly","ssam","gochujang","garlic"], 30),
        ("Tteokbokki", ["rice cakes","gochujang","fish cake","scallions"], 20),
        ("Doenjang Jjigae", ["fermented soybean paste","tofu","zucchini","clams"], 25),
        ("Haemul Pajeon", ["seafood","spring onion pancake","dipping sauce"], 20),
        ("Naengmyeon", ["buckwheat noodles","cold broth","cucumber","pickled radish"], 15),
        ("Bulgogi", ["marinated beef","pear","soy","sesame","lettuce wraps"], 30),
        ("Bingsu", ["shaved ice","red bean","condensed milk","tteok"], 10),
        ("Hobakjuk", ["pumpkin porridge","glutinous rice balls","ginger"], 40),
        ("Mandu", ["pork and chive dumplings","ginger","soy dipping sauce"], 30),
    ],
    "Middle Eastern": [
        ("Shawarma", ["chicken","garlic sauce","pita","pickles","sumac"], 20),
        ("Lamb Kofta", ["minced lamb","herbs","spices","flatbread","tzatziki"], 25),
        ("Mansaf", ["lamb","jameed sauce","rice","almonds","parsley"], 120),
        ("Fatteh", ["pita chips","chickpeas","yoghurt","tahini","pine nuts"], 20),
        ("Kibbeh", ["bulgur","lamb mince","pine nuts","cinnamon"], 45),
        ("Musakhan", ["roasted chicken","sumac","caramelised onions","taboon bread"], 60),
        ("Mahshi", ["stuffed vegetables","rice","herbs","tomato broth"], 75),
        ("Kunafa", ["shredded pastry","cheese","sugar syrup","rose water"], 45),
        ("Mutabbel", ["roasted aubergine","tahini","lemon","pomegranate"], 30),
        ("Arayes", ["minced lamb","herbs","pita","grilled"], 20),
        ("Qatayef", ["stuffed pancakes","cheese","walnuts","sugar syrup"], 30),
        ("Knafeh", ["vermicelli","cheese","syrup","pistachios"], 40),
        ("Freekeh Soup", ["freekeh","lamb","onion","allspice"], 60),
        ("Ma'amoul", ["semolina","date filling","rose water","powdered sugar"], 60),
        ("Shorbet Adas", ["red lentil soup","cumin","lemon","crispy pita"], 30),
    ],
    "Spanish": [
        ("Paella Valenciana", ["bomba rice","chicken","rabbit","green beans","saffron"], 60),
        ("Patatas Bravas", ["fried potatoes","spicy tomato sauce","aioli"], 30),
        ("Gazpacho", ["tomatoes","cucumber","peppers","sherry vinegar","olive oil"], 15),
        ("Tortilla Española", ["eggs","potatoes","onion","olive oil"], 30),
        ("Pulpo a la Gallega", ["octopus","paprika","olive oil","sea salt","potato"], 60),
        ("Jamón Ibérico Board", ["ibérico ham","manchego","olives","bread"], 5),
        ("Croquetas", ["béchamel","jamón","breadcrumbs","fried"], 45),
        ("Salmorejo", ["tomatoes","bread","garlic","olive oil","boiled egg"], 15),
        ("Fideuà", ["short noodles","seafood","alioli","saffron broth"], 40),
        ("Pisto Manchego", ["courgette","peppers","tomatoes","egg"], 35),
        ("Bravas de Morunos", ["spiced pork skewers","pimentón","lemon"], 20),
        ("Churros con Chocolate", ["fried dough","thick hot chocolate"], 20),
        ("Crema Catalana", ["cream","egg yolks","cinnamon","caramel crust"], 30),
        ("Ensalada Rusa", ["potato","carrot","peas","mayo","tuna"], 20),
        ("Cocido Madrileño", ["chickpeas","pork","chorizo","vegetables"], 180),
    ],
    "Greek": [
        ("Souvlaki Platter", ["pork skewers","pita","tzatziki","salad"], 30),
        ("Pastitsio", ["pasta","beef mince","béchamel","tomatoes"], 75),
        ("Loukoumades", ["fried dough balls","honey","cinnamon","walnuts"], 20),
        ("Fasolada", ["white bean soup","tomatoes","celery","olive oil"], 60),
        ("Kleftiko", ["slow-roasted lamb","garlic","lemon","feta","oregano"], 180),
        ("Saganaki", ["fried cheese","lemon","ouzo flambé"], 10),
        ("Avgolemono", ["chicken broth","egg","lemon","orzo"], 30),
        ("Gigantes Plaki", ["giant butter beans","tomato sauce","herbs","olive oil"], 60),
        ("Taramosalata", ["fish roe","bread","olive oil","lemon"], 10),
        ("Loukaniko", ["spiced pork sausage","orange peel","grilled"], 20),
        ("Revithada", ["slow-cooked chickpeas","rosemary","lemon","olive oil"], 120),
        ("Galaktoboureko", ["custard","filo pastry","syrup","lemon"], 60),
        ("Patsas", ["tripe soup","garlic vinegar","paprika"], 120),
        ("Briam", ["roasted vegetables","tomatoes","herbs","olive oil"], 60),
        ("Tsoureki", ["sweet braided bread","mahlab","mastic","eggs"], 180),
    ],
    "Vietnamese": [
        ("Pho Bo", ["beef broth","rice noodles","rare beef","herbs","bean sprouts"], 180),
        ("Banh Mi", ["baguette","pork","pâté","pickled daikon","jalapeño","cilantro"], 20),
        ("Bun Bo Hue", ["spicy beef broth","rice noodles","lemongrass","shrimp paste"], 90),
        ("Goi Cuon", ["rice paper","shrimp","pork","vermicelli","herbs"], 20),
        ("Com Tam", ["broken rice","grilled pork chop","cha trung","nuoc cham"], 20),
        ("Bun Cha", ["grilled pork patties","rice noodles","dipping broth","herbs"], 30),
        ("Cao Lau", ["rice noodles","pork","bean sprouts","crispy croutons"], 30),
        ("Mi Quang", ["turmeric noodles","shrimp","pork","peanuts","rice crackers"], 30),
        ("Che Ba Mau", ["three colour dessert","mung bean","coconut milk","jelly"], 20),
        ("Banh Xeo", ["crispy crepe","pork","shrimp","bean sprouts","mint"], 25),
        ("Bo La Lot", ["beef in betel leaves","lemongrass","grilled"], 25),
        ("Canh Chua", ["sour fish soup","tamarind","pineapple","tomatoes"], 30),
        ("Xoi Gac", ["red glutinous rice","gac fruit","chicken","mung beans"], 45),
        ("Banh Cuon", ["steamed rice rolls","pork","wood ear mushrooms","crispy shallots"], 30),
        ("Nuoc Cham", ["fish sauce","lime","sugar","chilli","garlic"], 5),
    ],
    "Ethiopian": [
        ("Doro Wat", ["chicken","berbere","niter kibbeh","hard-boiled egg","injera"], 90),
        ("Misir Wat", ["red lentils","berbere","onions","garlic","injera"], 45),
        ("Tibs", ["sautéed beef","rosemary","jalapeño","niter kibbeh"], 25),
        ("Shiro Wat", ["chickpea powder","berbere","garlic","injera"], 30),
        ("Atkilt Wat", ["cabbage","carrots","potato","turmeric","injera"], 30),
        ("Gored Gored", ["raw beef cubes","awaze","niter kibbeh"], 10),
        ("Kitfo", ["minced raw beef","mitmita","niter kibbeh","ayib"], 15),
        ("Injera", ["teff","water","fermented batter"], 60),
        ("Ayib", ["fresh cottage cheese","herbs"], 60),
        ("Kategna", ["injera","niter kibbeh","berbere","grilled"], 10),
        ("Firfir", ["torn injera","berbere sauce","niter kibbeh"], 15),
        ("Siga Tibs", ["beef strips","peppers","onions","rosemary"], 20),
        ("Atakilt", ["mixed vegetable stew","cabbage","potato","carrot"], 30),
        ("Buna", ["Ethiopian coffee ceremony","cardamom","popcorn"], 30),
        ("Sambusa", ["lentil-filled pastry","jalapeño","fried"], 30),
    ],
    "Peruvian": [
        ("Ceviche Clasico", ["fresh fish","lime juice","ají amarillo","red onion","cilantro"], 20),
        ("Lomo Saltado", ["beef strips","tomatoes","peppers","soy sauce","fries"], 20),
        ("Causa Limeña", ["potato","ají amarillo","chicken","avocado","lime"], 30),
        ("Ají de Gallina", ["chicken","ají amarillo","bread","walnuts","cream"], 45),
        ("Anticuchos", ["beef heart","cumin","ají panca","vinegar","grilled"], 30),
        ("Arroz con Leche", ["rice","milk","cinnamon","condensed milk","raisins"], 40),
        ("Pollo a la Brasa", ["roasted chicken","ají amarillo marinade","fries"], 90),
        ("Seco de Cordero", ["lamb","cilantro sauce","beer","chicha de jora"], 75),
        ("Papa a la Huancaína", ["boiled potatoes","huancaína sauce","olives","egg"], 20),
        ("Picarones", ["sweet potato doughnuts","chancaca syrup","anise"], 30),
        ("Tallarin Saltado", ["stir-fried noodles","beef","tomato","soy","peppers"], 20),
        ("Rocoto Relleno", ["stuffed rocoto pepper","beef mince","cheese","baked"], 60),
        ("Sudado de Pescado", ["fish","tomatoes","onions","ají amarillo","chicha"], 30),
        ("Chupe de Camarones", ["prawn chowder","potato","corn","ají panca","cream"], 45),
        ("Mazamorra Morada", ["purple corn pudding","dried fruits","cinnamon"], 40),
    ],
    "Turkish": [
        ("Döner Kebab", ["lamb","tomatoes","onions","flatbread","yoghurt sauce"], 30),
        ("Iskender Kebab", ["döner meat","tomato sauce","pide bread","browned butter"], 20),
        ("Manti", ["beef dumplings","garlic yoghurt","paprika butter"], 60),
        ("Imam Bayildi", ["stuffed aubergine","onions","tomatoes","olive oil"], 60),
        ("Börek", ["filo pastry","white cheese","spinach","egg"], 45),
        ("Mercimek Çorbası", ["red lentil soup","cumin","paprika","lemon"], 30),
        ("Köfte", ["minced lamb","parsley","cumin","grilled"], 20),
        ("Pilav", ["rice","butter","vermicelli","chicken stock"], 25),
        ("Baklava", ["filo","walnuts or pistachios","honey syrup","butter"], 90),
        ("Menemen", ["eggs","tomatoes","peppers","white cheese"], 15),
        ("Karnıyarık", ["aubergine","minced beef","tomatoes","peppers"], 60),
        ("Sütlaç", ["rice pudding","milk","sugar","cinnamon","rose water"], 45),
        ("Adana Kebab", ["spiced ground lamb","chilli","grilled on skewer"], 20),
        ("Gözleme", ["flatbread","cheese and spinach filling","pan-fried"], 15),
        ("Künefe", ["shredded pastry","soft cheese","sugar syrup","pistachios"], 20),
    ],
    "Moroccan": [
        ("Lamb Tagine", ["lamb","preserved lemon","olives","ras el hanout","prunes"], 120),
        ("Couscous Royale", ["semolina couscous","seven vegetables","merguez","harissa"], 60),
        ("Harira", ["tomatoes","lentils","chickpeas","lamb","cilantro"], 60),
        ("Bastilla", ["pigeon or chicken","almonds","filo pastry","cinnamon sugar"], 90),
        ("Mechoui", ["slow-roasted whole lamb","cumin","coriander","flatbread"], 240),
        ("Kefta Mkaouara", ["meatballs","tomato sauce","eggs","cumin"], 30),
        ("Zaalouk", ["smoky aubergine","tomatoes","garlic","cumin","olive oil"], 30),
        ("Chebakia", ["honey sesame cookies","orange blossom water","fried"], 60),
        ("Maaqouda", ["potato fritters","garlic","cumin","fresh herbs","egg"], 30),
        ("Briouat", ["crispy stuffed pastries","almond or cheese","honey"], 45),
        ("Rfissa", ["chicken","lentils","msemen bread","fenugreek"], 90),
        ("Chermoula Fish", ["marinated fish","paprika","cumin","cilantro","lemon"], 30),
        ("Sellou", ["toasted flour","almonds","sesame","honey","cinnamon"], 30),
        ("Msemen", ["square flatbread","butter","honey","served with tea"], 30),
        ("Mint Tea", ["green tea","fresh mint","sugar","poured ceremonially"], 10),
    ],
    "Lebanese": [
        ("Chicken Shawarma", ["chicken","garlic sauce","sumac","pita","pickles"], 20),
        ("Kibbeh Nayyeh", ["raw lamb","bulgur","pine nuts","spices","mint"], 20),
        ("Fattoush Salad", ["mixed greens","pita chips","sumac dressing","radish"], 15),
        ("Kafta Meshwi", ["grilled lamb kafta","onions","parsley","flatbread"], 20),
        ("Mujaddara", ["lentils","rice","caramelised onions","cumin"], 45),
        ("Lahm Baajin", ["lamb minced pizza","tomatoes","onions","parsley"], 30),
        ("Warak Enab", ["stuffed vine leaves","rice","lemon","olive oil"], 60),
        ("Sawda Djej", ["chicken livers","pomegranate molasses","garlic"], 15),
        ("Samke Harra", ["whole fish","tahini","pine nuts","harissa"], 45),
        ("Sfeeha", ["lamb-filled pastry","onions","pomegranate","pine nuts"], 45),
        ("Riz a Djej", ["chicken with rice","cinnamon","allspice","fried nuts"], 60),
        ("Arayess", ["kafta-filled pita","grilled","yoghurt dip"], 20),
        ("Mamoul", ["semolina cookies","date or nut filling","rose water"], 60),
        ("Ashta", ["clotted cream","orange blossom","rose water","pistachios"], 30),
        ("Jallab", ["grape juice","rose water","pomegranate","pine nuts"], 5),
    ],
    "Brazilian": [
        ("Feijoada", ["black beans","pork ribs","sausage","rice","farofa"], 180),
        ("Pão de Queijo", ["tapioca flour","cheese","egg","milk"], 30),
        ("Churrasco", ["picanha","sausage","chicken hearts","grilled skewers"], 60),
        ("Moqueca Baiana", ["fish","coconut milk","dendê oil","tomatoes","peppers"], 45),
        ("Coxinha", ["chicken-filled dough","shredded chicken","cream cheese","fried"], 45),
        ("Acarajé", ["black-eyed pea fritters","vatapá","caruru","shrimp"], 60),
        ("Brigadeiro", ["condensed milk","cocoa","butter","chocolate sprinkles"], 20),
        ("Picanha", ["top sirloin cap","rock salt","chimichurri"], 30),
        ("Caldinho de Feijão", ["black bean broth","garlic","cumin","farofa"], 30),
        ("Bobó de Camarão", ["shrimp","cassava purée","coconut milk","dendê"], 45),
        ("Pastel", ["fried pastry","cheese or beef filling","hot sauce"], 30),
        ("Quindim", ["egg yolk","sugar","shredded coconut","butter"], 45),
        ("Caipirinha", ["cachaça","lime","sugar","crushed ice"], 5),
        ("Tapioca", ["tapioca crêpe","coconut","banana or cheese filling"], 10),
        ("Pudim de Leite", ["condensed milk","egg","milk","caramel"], 60),
    ],
    "British": [
        ("Fish and Chips", ["cod","mushy peas","chips","malt vinegar","tartare"], 30),
        ("Full English Breakfast", ["bacon","eggs","sausage","baked beans","black pudding"], 20),
        ("Shepherd's Pie", ["lamb mince","peas","carrots","mashed potato topping"], 60),
        ("Beef Wellington", ["beef fillet","mushroom duxelles","prosciutto","puff pastry"], 90),
        ("Chicken Tikka Masala", ["chicken","tikka sauce","cream","naan"], 40),
        ("Sticky Toffee Pudding", ["dates","sponge","toffee sauce","vanilla ice cream"], 45),
        ("Cornish Pasty", ["beef skirt","potato","swede","onion","pastry"], 60),
        ("Bangers and Mash", ["pork sausages","mashed potato","onion gravy"], 30),
        ("Lancashire Hotpot", ["lamb","potato topping","kidneys","onions"], 120),
        ("Eton Mess", ["strawberries","meringue","whipped cream"], 15),
        ("Scotch Egg", ["hard-boiled egg","sausage meat","breadcrumbs","fried"], 30),
        ("Welsh Rarebit", ["cheese sauce","mustard","ale","toast"], 15),
        ("Pork Pie", ["hot-water crust pastry","pork filling","jelly"], 120),
        ("Toad in the Hole", ["sausages","Yorkshire pudding batter","onion gravy"], 45),
        ("Victoria Sponge", ["butter","sugar","eggs","jam","whipped cream"], 45),
    ],
    "German": [
        ("Bratwurst", ["pork sausage","mustard","sauerkraut","bretzel"], 15),
        ("Sauerbraten", ["marinated beef","red wine vinegar","gingersnap gravy"], 180),
        ("Schnitzel", ["veal or pork","breadcrumbs","lemon","potato salad"], 20),
        ("Spätzle", ["egg noodles","caramelised onions","gruyère","chives"], 20),
        ("Kartoffelsuppe", ["potato soup","leek","bacon","parsley"], 45),
        ("Black Forest Cake", ["chocolate sponge","cherries","kirsch","cream"], 90),
        ("Pretzels", ["bread dough","lye solution","coarse salt"], 60),
        ("Rouladen", ["beef rolls","mustard","bacon","pickles","onion gravy"], 120),
        ("Lebkuchen", ["gingerbread","honey","spices","chocolate coating"], 30),
        ("Kartoffelpuffer", ["potato pancakes","sour cream or apple sauce"], 20),
        ("Flammkuchen", ["thin pastry","crème fraîche","onion","lardons"], 15),
        ("Königsberger Klopse", ["veal meatballs","capers","cream sauce"], 45),
        ("Currywurst", ["bratwurst","curry ketchup","curry powder","fries"], 15),
        ("Apfelstrudel", ["apple","cinnamon","raisins","filo pastry","icing sugar"], 60),
        ("Rindfleischsuppe", ["beef broth","root vegetables","marrow","noodles"], 90),
    ],
    "Indonesian": [
        ("Nasi Goreng", ["fried rice","kecap manis","fried egg","shrimp","vegetables"], 15),
        ("Rendang", ["slow-cooked beef","coconut milk","lemongrass","galangal","chilli"], 180),
        ("Satay Ayam", ["chicken skewers","peanut sauce","ketupat","pickles"], 30),
        ("Gado Gado", ["vegetables","tofu","tempeh","peanut sauce","prawn crackers"], 20),
        ("Soto Ayam", ["chicken broth","turmeric","noodles","bean sprouts"], 45),
        ("Mie Goreng", ["fried egg noodles","chicken","egg","cabbage","kecap manis"], 15),
        ("Rawon", ["black beef soup","kluwek","bean sprouts","salted egg"], 120),
        ("Tempe Orek", ["fried tempeh","sweet soy","chilli","galangal"], 20),
        ("Bubur Ayam", ["rice porridge","chicken","ginger","crispy shallots"], 30),
        ("Pempek", ["fish cake","egg inside","cuko sauce","cucumber"], 45),
        ("Ayam Bakar", ["grilled marinated chicken","kecap manis","galangal"], 90),
        ("Babi Guling", ["Balinese suckling pig","turmeric","lemongrass","spice paste"], 360),
        ("Martabak", ["stuffed savoury pancake","egg","minced meat","leek"], 20),
        ("Klepon", ["glutinous rice balls","palm sugar","coconut"], 30),
        ("Es Campur", ["shaved ice","coconut milk","jelly","jackfruit","condensed milk"], 10),
    ],
    "Filipino": [
        ("Adobo", ["chicken or pork","soy sauce","vinegar","bay leaf","garlic"], 45),
        ("Sinigang", ["pork ribs","tamarind broth","kangkong","radish","eggplant"], 45),
        ("Lechon Kawali", ["crispy pork belly","liver sauce","sukang pinakurat"], 60),
        ("Kare-Kare", ["oxtail","peanut sauce","bok choy","banana blossom","bagoong"], 180),
        ("Sisig", ["pork cheek","liver","onion","chilli","egg","calamansi"], 30),
        ("Pancit Canton", ["egg noodles","pork","shrimp","vegetables","calamansi"], 20),
        ("Halo-Halo", ["shaved ice","ube","leche flan","beans","coconut","pinipig"], 15),
        ("Bulalo", ["beef bone marrow soup","corn","cabbage","peppercorns"], 180),
        ("Crispy Pata", ["deep-fried pork hock","vinegar dip","pickled papaya"], 120),
        ("Tinola", ["chicken","green papaya","chilli leaves","ginger broth"], 45),
        ("Bicol Express", ["pork","coconut milk","shrimp paste","lots of chilli"], 45),
        ("Lomi", ["thick egg noodles","pork","liver","quail eggs","thickened broth"], 30),
        ("Pinakbet", ["mixed vegetables","shrimp paste","pork","bitter melon"], 30),
        ("Biko", ["glutinous rice","coconut milk","palm sugar","latik"], 60),
        ("Leche Flan", ["egg yolks","condensed milk","evaporated milk","caramel"], 60),
    ],
    "Caribbean": [
        ("Jerk Chicken", ["chicken","scotch bonnet","allspice","thyme","rice and peas"], 90),
        ("Roti with Curry", ["curry duck or goat","flaky roti","potato","mango chutney"], 60),
        ("Oxtail Stew", ["oxtail","butter beans","scotch bonnet","thyme","allspice"], 180),
        ("Ackee and Saltfish", ["ackee","salted cod","scotch bonnet","onion","tomato"], 30),
        ("Pelau", ["rice","pigeon peas","chicken","caramelised sugar","coconut milk"], 45),
        ("Doubles", ["bara bread","channa","tamarind","pepper sauce","cucumber"], 20),
        ("Conch Fritters", ["conch","peppers","herbs","batter","hot sauce"], 20),
        ("Callaloo", ["dasheen leaves","coconut milk","okra","crab","scotch bonnet"], 30),
        ("Rice and Peas", ["jasmine rice","kidney beans","coconut milk","thyme"], 30),
        ("Festival", ["fried cornmeal dumplings","jerk chicken side"], 20),
        ("Saltfish Fritters", ["salted cod","spring onion","scotch bonnet","fried"], 20),
        ("Bake and Shark", ["fried bread","shark fillet","chutney","tamarind sauce"], 30),
        ("Rum Cake", ["dark rum","cherries","raisins","butter cake"], 90),
        ("Plantain Tostones", ["twice-fried green plantain","garlic mojo"], 15),
        ("Sorrel Drink", ["dried hibiscus","ginger","cloves","rum","sugar"], 15),
    ],
    "Russian": [
        ("Beef Stroganoff", ["beef strips","mushrooms","sour cream","egg noodles"], 30),
        ("Borscht", ["beetroot","cabbage","beef","sour cream","dill"], 90),
        ("Pelmeni", ["pork and beef dumplings","sour cream","butter","dill"], 45),
        ("Blini", ["thin crepes","sour cream","smoked salmon or caviar"], 20),
        ("Solyanka", ["mixed meat soup","pickles","olives","sour cream"], 60),
        ("Shashlik", ["marinated pork skewers","onion","tkemali sauce"], 60),
        ("Olivier Salad", ["potato","egg","pickles","peas","mayo"], 20),
        ("Syrniki", ["cottage cheese pancakes","sour cream","jam"], 20),
        ("Medovik", ["honey layer cake","sour cream frosting"], 120),
        ("Okroshka", ["cold kvas soup","potato","egg","cucumber","radish"], 15),
        ("Vatrushka", ["sweet bun","cottage cheese filling"], 90),
        ("Kotlety", ["pan-fried meat patties","mashed potato","pickles"], 30),
        ("Ukha", ["fish soup","root vegetables","dill","lemon"], 30),
        ("Kvass", ["fermented rye bread drink","raisins","lemon"], 1440),
        ("Napoleon Cake", ["puff pastry layers","pastry cream","caramel"], 180),
    ],
    "Polish": [
        ("Pierogi", ["potato and cheese dumplings","sour cream","fried onions"], 45),
        ("Bigos", ["hunter's stew","sauerkraut","mixed meats","mushrooms"], 120),
        ("Żurek", ["sour rye soup","hard-boiled egg","white sausage","marjoram"], 60),
        ("Kotlet Schabowy", ["breaded pork chop","cabbage salad","mashed potato"], 20),
        ("Gołąbki", ["stuffed cabbage rolls","pork and rice","tomato sauce"], 90),
        ("Barszcz", ["clear beetroot soup","uszka mushroom dumplings"], 60),
        ("Kapuśniak", ["sauerkraut soup","pork ribs","onion","caraway"], 75),
        ("Placki Ziemniaczane", ["potato pancakes","sour cream","mushroom sauce"], 30),
        ("Sernik", ["cheesecake","quark cheese","raisins","vanilla"], 90),
        ("Makowiec", ["poppy seed roll","honey","almonds","orange peel"], 120),
        ("Fasolka po Bretońsku", ["white beans","bacon","tomato sauce","marjoram"], 60),
        ("Chłodnik", ["cold beetroot soup","kefir","cucumber","dill","hard egg"], 10),
        ("Kielbasa", ["grilled Polish sausage","mustard","sauerkraut"], 15),
        ("Naleśniki", ["crepes","cottage cheese and vanilla filling","jam"], 20),
        ("Racuchy", ["apple fritters","powdered sugar","sour cream"], 20),
    ],
    "Nigerian": [
        ("Jollof Rice", ["long grain rice","tomato stew","chicken","peppers"], 60),
        ("Egusi Soup", ["ground melon seeds","palm oil","leafy greens","assorted meats"], 60),
        ("Puff Puff", ["fried dough balls","nutmeg","vanilla","sugar"], 30),
        ("Suya", ["spiced grilled beef skewers","yaji spice","onion","tomato"], 30),
        ("Pepper Soup", ["goat meat","utazi leaf","scent leaf","chilli broth"], 45),
        ("Moi Moi", ["steamed bean pudding","fish","egg","peppers"], 60),
        ("Eba with Egusi", ["cassava flour dough","egusi soup","stockfish"], 20),
        ("Banga Soup", ["palm nut soup","oxtail","crayfish","herbs"], 90),
        ("Ogbono Soup", ["wild mango seeds","palm oil","stockfish","spinach"], 45),
        ("Akara", ["fried black-eyed pea fritters","peppers","onion"], 20),
        ("Ofada Rice", ["local parboiled rice","ofada stew","assorted meats"], 45),
        ("Nkwobi", ["spiced cow foot","ugba seeds","palm oil","utazi"], 60),
        ("Fried Plantain", ["ripe plantain","salt","palm oil or vegetable oil"], 10),
        ("Ofe Akwu", ["palm nut stew","stockfish","assorted meats","uziza"], 60),
        ("Chin Chin", ["fried dough snack","coconut","nutmeg","sugar"], 30),
    ],
    "Argentinian": [
        ("Asado", ["beef ribs","chorizo","morcilla","chimichurri","provoleta"], 120),
        ("Empanadas", ["beef pastry parcels","egg","olives","raisins"], 45),
        ("Milanesa", ["breaded beef","mashed potato or fries","lemon"], 20),
        ("Locro", ["corn and bean stew","pork","chorizo","squash"], 120),
        ("Dulce de Leche Crepes", ["crepes","dulce de leche","whipped cream"], 20),
        ("Choripán", ["chorizo sandwich","chimichurri","crusty roll"], 15),
        ("Medialunas", ["croissant-style pastry","dulce de leche","glazed"], 45),
        ("Provoleta", ["grilled provolone","oregano","chilli flakes","bread"], 10),
        ("Revuelto Gramajo", ["scrambled eggs","ham","peas","fried potato strips"], 15),
        ("Torta Frita", ["fried flatbread","mate tea","dulce de leche"], 15),
        ("Carbonada Criolla", ["beef stew","corn","squash","pear","peach"], 90),
        ("Alfajores", ["cornstarch cookies","dulce de leche","desiccated coconut"], 45),
        ("Matambre Arrollado", ["rolled stuffed beef","hard egg","vegetables"], 90),
        ("Vitel Toné", ["cold veal","tuna mayo sauce","capers","anchovies"], 60),
        ("Submarino", ["hot milk","chocolate bar melted in","served with medialunas"], 5),
    ],
    "Swedish": [
        ("Swedish Meatballs", ["pork and beef mince","cream sauce","lingonberry","mashed potato"], 45),
        ("Gravlax", ["cured salmon","dill","mustard sauce","rye bread"], 1440),
        ("Smörgåsbord", ["assorted open sandwiches","herring","cheese","eggs"], 30),
        ("Janssons Frestelse", ["potato gratin","anchovies","cream","onion"], 60),
        ("Ärtsoppa", ["yellow pea soup","ham","mustard","served Thursdays"], 120),
        ("Kalops", ["beef stew","allspice","carrot","onion","bay leaf"], 90),
        ("Raggmunk", ["potato pancakes","lingonberry jam","fried pork"], 30),
        ("Kladdkaka", ["gooey chocolate cake","icing sugar","vanilla ice cream"], 25),
        ("Semla", ["cardamom bun","almond paste","whipped cream"], 90),
        ("Pyttipanna", ["Swedish hash","diced potato","beetroot","fried egg"], 20),
        ("Surströmming", ["fermented herring","flatbread","sour cream","onion"], 10),
        ("Kanelbullar", ["cinnamon buns","cardamom","pearl sugar"], 120),
        ("Husmanskost", ["traditional beef stew","root vegetables","dill"], 90),
        ("Toast Skagen", ["prawn toast","dill mayo","bleak roe","lemon"], 10),
        ("Prinsesstårta", ["princess cake","marzipan","custard","cream","sponge"], 120),
    ],
}

# Adjective and descriptor pools for name variation
ADJECTIVES = [
    "Spicy", "Creamy", "Smoky", "Crispy", "Tangy", "Hearty", "Zesty",
    "Fragrant", "Rustic", "Classic", "Modern", "Slow-Cooked", "Grilled",
    "Baked", "Chilled", "Pan-Fried", "Roasted", "Steamed", "Braised",
    "Double-Fried", "Smoked", "Glazed", "Stuffed", "Layered",
    "Homestyle", "Gourmet", "Street-Style", "Chef's", "Traditional",
    "Deconstructed", "Loaded", "Extra-Crispy", "Velvety", "Light",
]

PLATING_STYLES = [
    "served in a clay pot", "plated on banana leaf", "served in a bowl",
    "in a cast-iron skillet", "on a wooden board", "with a side of rice",
    "topped with fresh herbs", "with a drizzle of chilli oil",
    "garnished with microgreens", "with crusty bread on the side",
    "in a stone mortar", "plated elegantly", "served family-style",
    "with a yoghurt swirl", "garnished with pomegranate seeds",
    "with a squeeze of lime", "on a bed of greens", "with pickled vegetables",
    "accompanied by flatbread", "with toasted nuts sprinkled over",
]

MOOD_DESCRIPTIONS = {
    "happy":       ["uplifting", "bright", "celebratory", "joyful", "light"],
    "sad":         ["comforting", "warming", "soothing", "nostalgic", "gentle"],
    "stressed":    ["calming", "indulgent", "familiar", "reliable", "grounding"],
    "tired":       ["energising", "easy", "nourishing", "reviving", "simple"],
    "romantic":    ["elegant", "rich", "sensual", "indulgent", "intimate"],
    "excited":     ["bold", "vibrant", "festive", "adventurous", "fun"],
    "cozy":        ["warming", "hearty", "comforting", "snug", "familiar"],
    "adventurous": ["exotic", "bold", "unusual", "complex", "thrilling"],
    "anxious":     ["gentle", "mild", "calming", "soothing", "familiar"],
    "bored":       ["surprising", "interesting", "flavourful", "stimulating", "unique"],
    "nostalgic":   ["classic", "traditional", "homestyle", "reminiscent", "timeless"],
    "celebratory": ["festive", "indulgent", "showstopping", "luxurious", "special"],
    "lonely":      ["comforting", "warming", "simple", "honest", "nourishing"],
    "energetic":   ["fresh", "light", "vibrant", "protein-rich", "zingy"],
    "sluggish":    ["energising", "spiced", "stimulating", "reviving", "bold"],
    "focused":     ["clean", "light", "brain-boosting", "sustaining", "simple"],
    "heartbroken": ["comforting", "indulgent", "nostalgic", "gentle", "soothing"],
    "proud":       ["celebratory", "elaborate", "showstopping", "impressive", "rich"],
    "nervous":     ["mild", "familiar", "simple", "calming", "light"],
    "content":     ["satisfying", "balanced", "wholesome", "pleasant", "harmonious"],
}

DESCRIPTION_TEMPLATES = [
    "A {adj} {dish} from {cuisine} cuisine, perfect for when you're feeling {mood_desc}. "
    "Made with {ing1} and {ing2}, {plating}. Takes about {time} minutes to prepare.",

    "{dish} is a beloved {cuisine} classic — {adj} and deeply satisfying. "
    "The combination of {ing1} with {ing2} creates something truly special, {plating}. "
    "Ready in {time} minutes.",

    "This {adj} version of {dish} brings the best of {cuisine} cooking to your table. "
    "A {mood_desc} dish featuring {ing1} and {ing2}, {plating}. Prep time: {time} minutes.",

    "Treat yourself to this {adj} {dish}, a cornerstone of {cuisine} cuisine. "
    "With {ing1} and {ing2}, it's a {mood_desc} meal that comes together in {time} minutes, "
    "{plating}.",

    "Craving something {mood_desc}? This {cuisine} {dish} delivers. "
    "{adj} and full of character, it showcases {ing1} alongside {ing2}. "
    "{plating}. Perfect for a {time}-minute cook.",

    "A {mood_desc} {adj} {dish} rooted in authentic {cuisine} tradition. "
    "{ing1} and {ing2} are the stars, {plating}. On the table in {time} minutes.",

    "Few things hit quite like a {adj} {dish}. This {cuisine} staple is a "
    "{mood_desc} crowd-pleaser made with {ing1} and {ing2}, {plating}.",

    "This {cuisine} {dish} is what {mood_desc} eating looks like. "
    "{adj} in texture, generous with {ing1}, and finished with {ing2} — {plating}. "
    "About {time} minutes from start to finish.",
]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def pick_moods(n_primary=3, n_secondary=2):
    primary   = random.sample(MOODS_LIST, k=min(n_primary, len(MOODS_LIST)))
    secondary = random.sample(
        [m for m in MOODS_LIST if m not in primary],
        k=min(n_secondary, len(MOODS_LIST) - len(primary))
    )
    return primary + secondary

def pick_dietary(cuisine):
    """Assign realistic dietary tags based on cuisine and random chance."""
    tags = []
    roll = random.random()
    if roll < 0.25:
        tags.append("Vegetarian")
        if random.random() < 0.4:
            tags.append("Vegan")
    if random.random() < 0.18:
        tags.append("Gluten-Free")
    if random.random() < 0.15:
        tags.append("Dairy-Free")
    if random.random() < 0.10:
        tags.append("Nut-Free")
    if cuisine in ("Middle Eastern", "Lebanese", "Turkish", "Moroccan", "Nigerian"):
        if random.random() < 0.5:
            tags.append("Halal")
    if random.random() < 0.12:
        tags.append("High-Protein")
    if random.random() < 0.08:
        tags.append("Low-Carb")
    if random.random() < 0.06:
        tags.append("Keto")
    return list(set(tags))

def build_description(dish_name, cuisine, ingredients, prep_time, moods):
    mood_key   = moods[0] if moods else "happy"
    mood_descs = MOOD_DESCRIPTIONS.get(mood_key, ["delicious"])
    mood_desc  = random.choice(mood_descs)
    adj        = random.choice(ADJECTIVES).lower()
    plating    = random.choice(PLATING_STYLES)
    ing_list   = [i.strip() for i in ingredients.split(",") if i.strip()]
    ing1 = ing_list[0] if len(ing_list) > 0 else "fresh ingredients"
    ing2 = ing_list[1] if len(ing_list) > 1 else "aromatic spices"
    template   = random.choice(DESCRIPTION_TEMPLATES)
    return template.format(
        adj=adj, dish=dish_name, cuisine=cuisine,
        mood_desc=mood_desc, ing1=ing1, ing2=ing2,
        plating=plating, time=prep_time,
    )

def make_name_variant(base_name, adj_chance=0.45):
    if random.random() < adj_chance:
        adj = random.choice(ADJECTIVES)
        return f"{adj} {base_name}"
    return base_name

# ─────────────────────────────────────────────────────────────
# GENERATE
# ─────────────────────────────────────────────────────────────

def generate_rows(target=50000):
    rows = []
    row_id = 1

    # Flatten all base dishes
    base_dishes = []
    for cuisine, dishes in CUISINE_DISHES.items():
        for dish in dishes:
            base_dishes.append((cuisine, dish))

    # Calculate how many variants per dish
    n_base      = len(base_dishes)           # 450 base dishes
    variants_per = target // n_base + 2      # ~112 variants each → well over 50k

    print(f"Base dishes: {n_base} | Target: {target} | Variants/dish: {variants_per}")

    for cuisine, (base_name, base_ingredients, base_time) in base_dishes:
        if len(rows) >= target:
            break
        for _ in range(variants_per):
            if len(rows) >= target:
                break

            # Name variation
            name = make_name_variant(base_name)

            # Ingredients — sometimes swap one
            ings = list(base_ingredients)
            if random.random() < 0.2 and len(ings) > 2:
                ings[random.randint(0, len(ings)-1)] = random.choice([
                    "seasonal herbs", "chilli flakes", "garlic", "lemon zest",
                    "smoked paprika", "coconut cream", "parmesan", "fresh ginger",
                    "roasted garlic", "caramelised onion", "bone broth", "tahini",
                ])
            ingredients_str = ", ".join(ings)

            # Prep time variation (±25%)
            prep_time = max(5, int(base_time * random.uniform(0.75, 1.25)))

            # Moods
            n_primary   = random.randint(2, 4)
            n_secondary = random.randint(1, 3)
            moods       = pick_moods(n_primary, n_secondary)
            moods_str   = ", ".join(moods)

            # Dietary
            dietary      = pick_dietary(cuisine)
            dietary_str  = ", ".join(dietary) if dietary else ""

            # Spice level
            spice = random.choice(["Mild", "Medium", "Spicy", "Very Spicy", "None"])

            # Meal type
            meal_type = random.choice([
                "Main Course", "Starter", "Dessert", "Snack",
                "Soup", "Salad", "Breakfast", "Side Dish", "Drink",
            ])

            # Occasion
            occasion = random.choice(OCCASIONS)

            # Cooking method
            method = random.choice(COOKING_METHODS)

            # Flavour profile
            flavour = random.choice(FLAVOUR_PROFILES)

            # Texture
            texture = random.choice(TEXTURES)

            # Calories estimate
            calories = random.randint(150, 950)

            # Serving size
            servings = random.randint(1, 6)

            # Description
            description = build_description(
                name, cuisine, ingredients_str, prep_time, moods
            )

            rows.append({
                "id":               f"dish_{row_id:06d}",
                "name":             name,
                "cuisine":          cuisine,
                "meal_type":        meal_type,
                "description":      description,
                "ingredients":      ingredients_str,
                "moods":            moods_str,
                "dietary_tags":     dietary_str,
                "spice_level":      spice,
                "prep_time_mins":   prep_time,
                "calories_approx":  calories,
                "servings":         servings,
                "cooking_method":   method,
                "flavour_profile":  flavour,
                "texture":          texture,
                "occasion":         occasion,
            })
            row_id += 1

    random.shuffle(rows)
    print(f"Generated {len(rows)} rows.")
    return rows

# ─────────────────────────────────────────────────────────────
# WRITE CSV
# ─────────────────────────────────────────────────────────────

def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved → {path}  ({len(rows):,} rows)")

if __name__ == "__main__":
    rows = generate_rows(1000)
    write_csv(rows, "C:\\Users\\manit\\OneDrive\\Desktop\\projects\\food-mood-rag-chatbot\\data\\raw\\food_dataset.csv")