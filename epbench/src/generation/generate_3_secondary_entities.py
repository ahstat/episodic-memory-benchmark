import random
import re
from tiktoken import get_encoding
from epbench.src.generation.raw_materials import parameters_universe_func
from epbench.src.generation.generate_1_events_and_meta_events import find_duplicates
from epbench.src.generation.printing import get_single_ei
from epbench.src.generation.verification_direct import cut_paragraph_func, remove_initial_numbering

def check_for_post_duplicates(parameters_universe, parameters_entity_fill):
    # alone
    duplicated_first_names = find_duplicates(parameters_universe['first_names'])
    duplicated_last_names = find_duplicates(parameters_universe['last_names'])
    if len(duplicated_first_names) > 0:
        print(duplicated_first_names)
        raise ValueError('Duplicated first names')
    if len(duplicated_last_names) > 0:
        print(duplicated_last_names)
        raise ValueError('Duplicated last names')
    
    # alone
    duplicated_last_names2 = find_duplicates(parameters_entity_fill['last_names_post'])
    if len(duplicated_last_names2) > 0:
        print(duplicated_last_names2)
        raise ValueError('Duplicated post last names')
    
    # together
    all_last_names = parameters_entity_fill['last_names_post'] + parameters_universe['last_names']
    duplicated_last_names3 = find_duplicates(all_last_names)
    if len(duplicated_last_names3) > 0:
        print(duplicated_last_names3)
        raise ValueError('Intersection between the two sets of last names is not 0')
    
    # check duplicates between first and last names (we allow identical first and last names for the original set)
    all_names = parameters_entity_fill['last_names_post'] + list(set(parameters_universe['last_names'] + parameters_universe['first_names']))
    if len(find_duplicates(all_names)) > 0:
        raise ValueError('Intersection between the new last names and either old first or last names is not empty')
    
    # check with first names
    all_names = parameters_entity_fill['first_names_post'] + list(set(parameters_universe['last_names'] + parameters_universe['first_names'] + parameters_entity_fill['last_names_post']))
    if len(find_duplicates(all_names)) > 0:
        raise ValueError('Intersection between the new first names and all the other names is not empty')

    return 0 

def generate_post_universe(nb_names = 2000, name_universe = 'default', seed_universe = 0):
    parameters_entity_fill = parameters_entity_fill_func(name_universe)
    parameters_universe = parameters_universe_func(name_universe)
    check_for_post_duplicates(parameters_universe, parameters_entity_fill)
    post_entities = generate_post_entities(nb_names, parameters_entity_fill['first_names_post'], parameters_entity_fill['last_names_post'], seed_universe)
    return post_entities

def find_all_but_duplicates(lst):
    seen = {}
    duplicates = []
    
    for item in lst:
        if item in seen:
            if seen[item] == 1:
                duplicates.append(item)
            seen[item] += 1
        else:
            seen[item] = 1
    return list(seen)

def helper_enumeration_to_list_func(to_convert = '''Empire State Building
Statue of Liberty
Museum of Modern Art
Chrysler Building
Wall Street
Woolworth Building
Trinity Church
Federal Hall
Mashomack Preserve
Orient Beach State Park'''):
    return to_convert.split('\n')

def difference(A, B):
    return list(set(A) - set(B))

def generate_post_entities(N, first_names, last_names, seed = 1):
    # Use for generating names for the post entities
    # not using the original one because too slow for N>30000
    # not changing the original one because it would change the seed behavior
    random.seed(seed+1)
    first_names_list = random.choices(first_names, k=5*N)
    random.seed(seed+2) # reinitialize the seed so that the final list is a subset when decreasing `N`
    last_names_list = random.choices(last_names, k=5*N)
    entities_with_rep = [f"{first} {last}" for (first, last) in zip(first_names_list, last_names_list)]
    entities = find_all_but_duplicates(entities_with_rep)
    if len(entities) < N:
        print(len(entities))
        raise ValueError('Too few post entities')
    return entities[:N]

def parameters_entity_fill_func(name_universe = 'default'):
    # Categories are: date, location, entity, content. That are assembled to build an event.

    if not name_universe.lower() == name_universe:
        raise ValueError('name_universe should be lower case')

    if (name_universe == 'default') or (name_universe == 'news') or (name_universe == 'scifi'):
        first_names_post = ['Asa', 'Dakari', 'Brodie', 'Finnley', 'Brinley', 'Timothy', 'Zachary', 'Blythe', 'Sasha', 'Alexa', 'Remy', 'Miracle', 'Rey', 'Milana', 'Meredith', 'Eli', 'Tatum', 'Alistair', 'Corey', 'Zaria', 'Matthew', 'Harleigh', 'Wesson', 'Uma', 'Nehemiah', 'Kenzie', 'Naya', 'Gwendolyn', 'Alvaro', 'Sylas', 'Blaze', 'Lachlan', 'Elsa', 'Skyla', 'Trey', 'Carson', 'Thatcher', 'Yousef', 'Mckenna', 'Ivory', 'Evangeline', 'Amara', 'Noor', 'Terry', 'Raquel', 'Yosef', 'Alondra', 'Esmeralda', 'Mackenzie', 'Aviana', 'Luella', 'Elijah', 'Zeke', 'Eddie', 'Shawn', 'Lennox', 'Tristan', 'Arden', 'Liliana', 'Madeleine', 'Octavia', 'Imran', 'Vivian', 'Cecilia', 'Denver', 'Waylon', 'Liv', 'Noa', 'Calla', 'Santino', 'Adelaide', 'Caspian', 'Everest', 'Yasmin', 'Ryker', 'Matthias', 'Ren', 'Averie', 'Jonathan', 'Ondine', 'Frankie', 'Dakota', 'Samir', 'Zuri', 'Greyson', 'Ainsley', 'Koa', 'Koda', 'Jeremiah', 'Naomi', 'Malaya', 'Lark', 'Nash', 'Konnor', 'Zephyr', 'Callum', 'Paris', 'Elsie', 'Paz', 'Ari', 'Braden', 'Sierra', 'Kaiden', 'Talia', 'Jensen', 'Jamir', 'Finnegan', 'Demi', 'Karsyn', 'Delphi', 'Presley', 'Reyna', 'Judah', 'Greta', 'Rocco', 'Nyla', 'Cora', 'Raphael', 'Beau', 'Journee', 'Ruth', 'Azriel', 'Eithan', 'Amaris', 'Gianluca', 'Dior', 'Maci', 'Reid', 'Lorelai', 'Sullivan', 'Blair', 'Julius', 'Sky', 'Arya', 'Ryleigh', 'Vienna', 'Kehlani', 'Ariadne', 'Theo', 'Blake', 'River', 'Ledger', 'Rosalie', 'Kiera', 'Quill', 'Bianca', 'Brynn', 'Zechariah', 'Milani', 'Gemma', 'Kyler', 'Ingrid', 'Gideon', 'Karina', 'Enzo', 'Jamal', 'Cade', 'Lana', 'Nicholas', 'Jayce', 'Hayden', 'Lev', 'Landen', 'Ada', 'Austin', 'Bryson', 'London', 'Charli', 'Malachi', 'Madisyn', 'Odette', 'Maryam', 'Camille', 'Messiah', 'Tadeo', 'Kaydence', 'Augustine', 'Eliza', 'Jorah', 'Eleanor', 'Marcel', 'Ahmed', 'Norah', 'Javion', 'Oaklynn', 'Ellery', 'Cian', 'Benson', 'Kaya', 'Nicole', 'Jonah', 'Callen', 'Seren', 'Cyrus', 'Bentlee', 'Naia', 'Vada', 'Heidi', 'Celeste', 'Mikayla', 'Serenity', 'Cairo', 'Dominick', 'Indigo', 'Maison', 'Ace', 'Kylan', 'Hamza', 'Halo', 'Raelynn', 'Kassidy', 'Luka', 'Braxton', 'Calvin', 'Alaric', 'Joshua', 'Hezekiah', 'Catherine', 'Kali', 'Vaughn', 'Caius', 'Ezekiel', 'Creed', 'Aura', 'Miles', 'Nala', 'Zayne', 'Haven', 'Jagger', 'Genevieve', 'Hugo', 'Summer', 'Kameron', 'Kamden', 'Ephraim', 'Hadley', 'Merrick', 'Abdiel', 'Lila', 'Khari', 'Zayn', 'Astrid', 'Easton', 'Annabelle', 'Angelina', 'Alicia', 'Caden', 'Ronan', 'Cillian', 'Kyson', 'Xander', 'Daisy', 'Pax', 'Charlie', 'Sincere', 'Selah', 'Anna', 'Bruno', 'Clara', 'Alayah', 'Zyair', 'Tessa', 'Daniela', 'Bridgette', 'Memphis', 'Wrenn', 'Rylie', 'Crew', 'Zoie', 'Colter', 'Zander', 'Darius', 'Dariel', 'Natalia', 'Emilia', 'Angelo', 'Cody', 'Delilah', 'Josephine', 'Brayden', 'Justin', 'Carly', 'Anthony', 'Celine', 'Frey', 'Mylah', 'Weston', 'Deandre', 'Salem', 'Miriam', 'Rohan', 'Briggs', 'Chaim', 'Evelynn', 'Kai', 'Poet', 'Kieran', 'Kora', 'Robert', 'Keanu', 'Tayo', 'Isabelle', 'Aaron', 'Catalina', 'Anders', 'Sadie', 'Urijah', 'Lyanna', 'Legend', 'Rowen', 'Achilles', 'Gracie', 'Scarlet', 'Remi', 'Issac', 'Ezequiel', 'Ukiah', 'Pike', 'Reginald', 'Maxton', 'Nola', 'Nylah', 'Inez', 'Henley', 'Abram', 'Amira', 'Autumn', 'Poppy', 'Banks', 'Wren', 'Kailani', 'Myla', 'Arianna', 'Nia', 'Markus', 'Thor', 'Beatrix', 'Daphne', 'Emory', 'Reign', 'Karter', 'Juelz', 'Gabriella', 'Joy', 'Sonny', 'Bear', 'Dante', 'Ila', 'Jenna', 'Ryland', 'Jair', 'Ronnie', 'Declan', 'Jolene', 'Brady', 'Damian', 'Elise', 'Stevie', 'Dylan', 'Una', 'Lilah', 'Khloe', 'Leila', 'Helen', 'Thea', 'Zev', 'Sunny', 'Mekhi', 'Lina', 'Matilda', 'Emersyn', 'Juliet', 'Natasha', 'Joziah', 'Zahra', 'Esther', 'Kyng', 'Isabela', 'Azael', 'August', 'Osvaldo', 'Sienna', 'Estrella', 'Marisol', 'Adan', 'Adonis', 'Elianna', 'Gael', 'Reagan', 'Gracelynn', 'Jeremias', 'Kendra', 'Kash', 'Juniper', 'Alma', 'Sariyah', 'Brianna', 'Jasper', 'Raven', 'Raina', 'Alaina', 'Chelsea', 'Alexis', 'Freya', 'Alec', 'Isla', 'Rhys', 'Macy', 'Mae', 'Avah', 'Dash', 'Zavier', 'Aubrie', 'Osiris', 'Kellan', 'Lea', 'Saoirse', 'Uri', 'Kylee', 'Fable', 'Brooklynn', 'Killian', 'Colson', 'Mara', 'Rowan', 'Gustavo', 'Sterling', 'Adrian', 'Talon', 'Sol', 'Seth', 'Gunner', 'Moshe', 'Samara', 'Enrique', 'Ophelia', 'Janelle', 'Kira', 'Gage', 'Colton', 'Hallie', 'Amir', 'Aarav', 'Jace', 'Maddison', 'Felix', 'Skye', 'Laurel', 'Shay', 'Roland', 'Diem', 'Rachel', 'Kaleb', 'Aileen', 'Lia', 'Thaddeus', 'Faye', 'Eden', 'Eloise', 'Madelyn', 'Alan', 'Madeline', 'Leighton', 'Harlow', 'Clover', 'Melody', 'Kellen', 'Mathias', 'Rylee', 'Oakley', 'Jett', 'Cleo', 'Axl', 'Kye', 'Raymond', 'Damir', 'Mckenzie', 'Michelle', 'Anika', 'Brielle', 'Milo', 'Brian', 'Zion', 'Athena', 'Lionel', 'Dion', 'Gus', 'Phoebe', 'Kylo', 'George', 'Millie', 'Mabel', 'Arian', 'Kenna', 'Archer', 'Marcelo', 'Israel', 'Aster', 'Kaden', 'Charlee', 'Ciara', 'Alden', 'Kairi', 'Case', 'Rebecca', 'Dexter', 'Adah', 'Nox', 'Kason', 'Camilla', 'Wesley', 'Viggo', 'Myles', 'Nadia', 'Joseph', 'Madison', 'Jude', 'Gia', 'Kyrie', 'Saylor', 'Kyla', 'Bronson', 'Darian', 'Emerald', 'Iris', 'Kase', 'Roan', 'Bastian', 'Finch', 'Gianna', 'Annie', 'Magnolia', 'Ruby', 'Mika', 'Boone', 'Ali', 'Gabriela', 'Charley', 'Emmett', 'Aisha', 'Tomas', 'Toby', 'Leia', 'Caiden', 'Holden', 'Grace', 'Xerxes', 'Marley', 'Atticus', 'Eliana', 'Ximena', 'Rio', 'Chaya', 'Cayson', 'Iker', 'Neymar', 'Malik', 'Huxley', 'Tori', 'Landry', 'Jaiden', 'Rune', 'Irene', 'Byron', 'Jaxton', 'Courtney', 'Briella', 'Rudy', 'Gianni', 'Adelyn', 'Kathryn', 'Sawyer', 'Graham', 'Kyro', 'Cash', 'Mercy', 'Xen', 'Aspen', 'Addison', 'Kadence', 'Ares', 'Jasiah', 'Anakin', 'Ansley', 'Elowen', 'Journey', 'Xara', 'Kennedi', 'Petra', 'Kace', 'Coraline', 'Maximus', 'Jaxx', 'Omar', 'Ford', 'Navy', 'Paul', 'Zaniyah', 'Opal', 'Amias', 'Emir', 'Ione', 'Ryder', 'Lexi', 'Soren', 'Victoria', 'Quentin', 'Casimir', 'Quinn', 'Barrett', 'Kayden', 'Kole', 'Trevor', 'Annika', 'Princeton', 'Romina', 'Orin', 'Callie', 'Makai', 'Emani', 'Aubree', 'Zelda', 'Julien', 'Michaela', 'Hope', 'Zakai', 'Lacey', 'Ike', 'Landon', 'Lyla', 'Jadiel', 'Wilder', 'Bowen', 'Elian', 'Arjun', 'Bodhi', 'Atlas', 'Leonel', 'Lawson', 'Colette', 'Juliana', 'Karson', 'Mccoy', 'Sage', 'Adelynn', 'Phoenix', 'Lyric', 'Juliette', 'Jax', 'Odin', 'Sylvia', 'Mariana', 'Lilith', 'Lennon', 'Nasir', 'Brittany', 'Bowie', 'Damon', 'Lena', 'Renata', 'Livia', 'Priscilla', 'Boden', 'Ronin', 'Olive', 'Jessa', 'Joel', 'Otto', 'Rory', 'Natalie', 'Harmony', 'Kingston', 'Kamila', 'Ariana', 'Ayla', 'Lucca', 'Kymani', 'Kaia', 'Willow', 'Elena', 'Elliot', 'Zyaire', 'Bode', 'Emmitt', 'Kinsley', 'Azariah', 'Taya', 'Tyson', 'Lilian', 'Kade', 'Rayden', 'Fleur', 'Vera', 'Bryn', 'Payton', 'Valentina', 'Raelyn', 'Zaid', 'Kobe', 'Tali', 'Angel', 'Melanie', 'Rayne', 'Adeline', 'Clarissa', 'Aylin', 'Anahi', 'Reese', 'Terrance', 'Hana', 'Ana', 'Rosemary', 'Georgia', 'Leonidas', 'Brody', 'Amina', 'Lucian', 'Jayla', 'Rosie', 'Molly', 'Drake', 'Jordyn', 'Giovanni', 'Joaquin', 'Winter', 'Colt', 'Zaiden', 'Dahlia', 'Juno', 'Zariah', 'Emelia', 'Brecken', 'Vincenzo', 'Vega', 'Anaya', 'Noelle', 'Khalil', 'Paulina', 'Xiomara', 'Orion', 'Aniyah', 'Adalyn', 'Leander', 'Amari', 'Lainey', 'Ula', 'Maria', 'Jericho', 'Zola', 'Harley', 'Persephone', 'Alison', 'Grey', 'Remington', 'Bjorn', 'Leandro', 'Veronica', 'Nico', 'Rodrigo', 'Deacon', 'Waverly', 'Malia', 'Fiona', 'Kynlee', 'Castiel', 'Malcolm', 'Ashton', 'Nekoda', 'Carlos', 'Saint', 'Jade', 'Braylen', 'Danna', 'Zane', 'Onyx', 'Jazlyn', 'Major', 'Amiyah', 'Stetson', 'Leilani', 'Amaia', 'Ivy', 'Nolan', 'Mira', 'Aurelia', 'Mina', 'Adriana', 'Axel', 'Sloane', 'Mac', 'Emiliano', 'Holland', 'Harlan', 'Madelynn', 'Aziel', 'Gaia', 'Beckett', 'Eva', 'Canaan', 'Yannick', 'Penny', 'Mauricio', 'Eben', 'Julia', 'Javier', 'Kensley', 'Lydia', 'Indie', 'Kenneth', 'Lumi', 'Kinslee', 'Thiago', 'Lola', 'Margot', 'Erin', 'Alana', 'Neo', 'Angela', 'Omari', 'Idris', 'Alice', 'Jared', 'Frida', 'Connor', 'Zain', 'Harmoni', 'Hendrix', 'Tiana', 'Uriah', 'Yara', 'Xavier', 'Cristian', 'Caroline', 'Maverick', 'Maeve', 'Finley', 'Maliyah', 'Dax', 'Hattie', 'Echo', 'Keira', 'Elodie', 'Niko', 'Hank', 'Maisie', 'Jayceon', 'Azrael', 'Vale', 'Atreus', 'Jameson', 'Bellamy', 'Casen', 'Dominic', 'Kian', 'Sylvie', 'Aaliyah', 'Fern', 'Nella', 'Scout', 'Elliott', 'Ivan', 'Quest', 'Lucia', 'Nathaniel', 'Sarah', 'Simone', 'Holly', 'Ian', 'Blaire', 'Kody', 'Silas', 'Kaylee', 'Paige', 'Rhett', 'Briar', 'Kolton', 'Jessie', 'Zora', 'Reina', 'Bridget', 'Liana', 'Mya', 'Keegan', 'Nixon', 'Cove', 'Thalia', 'Cynthia', 'Esperanza', 'Kalani', 'Iver', 'Elliana', 'Amiya', 'Yehuda', 'Shiloh', 'Amora', 'Jovie', 'Bonnie', 'Kaison', 'June', 'Cassius', 'Willa', 'Yuki', 'Halle', 'Ember', 'Heaven', 'Blakely', 'Zara', 'Leif', 'Vander', 'Finn', 'Selena', 'Rivka', 'Calliope', 'Nina', 'Xena', 'Simon', 'Arlo', 'Titan', 'Odelia', 'Darren', 'Maren', 'Arielle', 'Lorelei', 'Valeria', 'Evie']
        first_names_post = first_names_post[:500]
        # asked for 1500
        # 1387 only after removing duplicates
        # 1287 after removing the ones before
        # then cut the 1000 first
        last_names_post = ['Burleson', 'Maddox', 'Farnsworth', 'Henning', 'Aguilera', 'Frost', 'Burrell', 'Crawford', 'Corbin', 'Naquin', 'Marvin', 'Zeigler', 'Eastman', 'Cromwell', 'Richard', 'Baxter', 'Hailey', 'Franklin', 'Godwin', 'Desai', 'Olivier', 'Wiseman', 'Koon', 'Francois', 'Forrester', 'Cain', 'Blackwell', 'Davenport', 'Landers', 'Jenkins', 'Dickerson', 'Westbrook', 'Parrott', 'Medina', 'Christensen', 'Pierre', 'Atkinson', 'Whitmore', 'Webber', 'Rush', 'Harrigan', 'McDowell', 'Hacker', 'Drury', 'Meyers', 'Gilchrist', 'Sales', 'Morrow', 'Shepherd', 'Everett', 'Mickelson', 'Ritter', 'Overby', 'Knutson', 'Coke', 'Roman', 'Hamrick', 'Whittaker', 'Kelley', 'Dominguez', 'Macias', 'Piper', 'Blevins', 'Gifford', 'Schilling', 'Ybarra', 'Platt', 'Miranda', 'Pineda', 'Childs', 'Waller', 'Akers', 'McClure', 'Ziegler', 'Gallegos', 'Jordan', 'Faircloth', 'Badger', 'Ferris', 'Nichols', 'Ransom', 'Mcclintock', 'Langston', 'Meier', 'Blunt', 'Fritz', 'Cofer', 'Kirk', 'Barker', 'Cobb', 'Nicholson', 'Grossman', 'Rigby', 'Keel', 'Stern', 'Nesbitt', 'Woodruff', 'Emmert', 'Slater', 'Paxton', 'Diehl', 'Keeler', 'Boddie', 'Santana', 'Becker', 'Foote', 'Stahl', 'Peabody', 'Lachance', 'Salazar', 'Renteria', 'Hutchison', 'Fennell', 'Padgett', 'Holbrook', 'Krueger', 'Hamill', 'Gant', 'Clifford', 'Graves', 'Trimble', 'Hansen', 'Moses', 'Ferraro', 'Garvey', 'Whittle', 'Yang', 'Kinney', 'Walter', 'Mather', 'Luongo', 'Wessels', 'Dixon', 'Lozano', 'Grainger', 'McFadden', 'Bower', 'Ponce', 'Ghoston', 'Gordon', 'Hollis', 'Forsythe', 'Waldrop', 'Carver', 'Kendall', 'Delaney', 'Dubois', 'Petersen', 'Todd', 'Denton', 'Hathaway', 'Perkins', 'Gaylord', 'Novak', 'Spears', 'Keefe', 'Moran', 'Oliveira', 'Bean', 'Layman', 'Chávez', 'Bray', 'Foresman', 'Burrows', 'Castro', 'Noriega', 'Conley', 'Fitzgerald', 'Arrington', 'Stafford', 'Bradford', 'Matson', 'Witte', 'Starkey', 'Yaeger', 'Russ', 'Browning', 'Hostetler', 'Aguilar', 'Fernandez', 'Shinn', 'Waterman', 'Ely', 'Harkins', 'Rooney', 'Leclair', 'Adkins', 'Guenther', 'McClain', 'Borges', 'Sexton', 'Lynn', 'Trevino', 'Underhill', 'Hicks', 'Vaughan', 'Hardin', 'Pankey', 'Britt', 'Bruce', 'Kessler', 'Mclemore', 'Schaefer', 'Dobbins', 'Chase', 'Hoover', 'Woods', 'Wilcox', 'Vang', 'Brandt', 'Simonson', 'Phelps', 'Hunt', 'Putman', 'Faulkner', 'Angell', 'Arce', 'Vasquez', 'Armstrong', 'Keane', 'Lamb', 'Vest', 'Coker', 'Holcomb', 'Warner', 'Randolph', 'Fox', 'Dahl', 'Craine', 'Herrera', 'Grimes', 'Lowery', 'Sisco', 'Beasley', 'Diamond', 'Clifton', 'Newcomb', 'Tate', 'Barajas', 'Molina', 'Emery', 'Greenberg', 'Dawson', 'Pettit', 'Emerson', 'Mangum', 'Keefer', 'Pruitt', 'Willey', 'Freeze', 'Graff', 'Riddle', 'Kilpatrick', 'Quigley', 'Ryals', 'Weeks', 'Alaniz', 'Mccloskey', 'Snow', 'Matthews', 'Caudill', 'Stevens', 'McLean', 'Wynn', 'Schell', 'Alston', 'Stanton', 'Devore', 'Groth', 'Brunner', 'Stokes', 'Lantz', 'Robertson', 'Tennant', 'Leach', 'Rangel', 'Pham', 'Maggio', 'Delong', 'Shearer', 'Love', 'Short', 'Horn', 'Shipley', 'Leal', 'Stephenson', 'Pinto', 'Patterson', 'Crenshaw', 'Mays', 'Lehmann', 'Castaneda', 'Samson', 'Black', 'McCarthy', 'Bannister', 'Hinkle', 'Tunnell', 'Lam', 'Hoskins', 'Sweeney', 'Butler', 'Magee', 'Ferreira', 'Parham', 'Vanover', 'Faber', 'Acker', 'Aultman', 'Dunning', 'Brunson', 'Blaine', 'Rubin', 'Melvin', 'Manley', 'Hupp', 'Brito', 'Lutz', 'Kramer', 'Simmons', 'Lovell', 'Berry', 'Kennedy', 'Hardman', 'Ivey', 'Xiong', 'Wells', 'Fontaine', 'Mertz', 'Weatherly', 'Boyd', 'Espinoza', 'Gaines', 'Vo', 'Meyer', 'Foreman', 'Cunningham', 'Frye', 'Rowell', 'Godsey', 'Gaston', 'Lamm', 'Chang', 'Mercado', 'Garza', 'Duncan', 'Jeffers', 'Lara', 'Mack', 'Friedman', 'Griffin', 'Knight', 'Gable', 'Talley', 'Harrell', 'Reis', 'Hood', 'Kirkland', 'Erickson', 'Mathews', 'Freeman', 'Eaton', 'Ellison', 'Kuhn', 'Marlow', 'Rhoades', 'Herring', 'Fisher', 'Childress', 'Ragan', 'Chandler', 'Neumann', 'Dunham', 'Burdette', 'Johannsen', 'Keith', 'Stamper', 'Perry', 'Curley', 'Snell', 'Schaeffer', 'Cahill', 'Bradley', 'Segura', 'Maher', 'Skutt', 'Golden', 'Rader', 'Hellman', 'Soto', 'Bentley', 'Pennington', 'Callahan', 'Liebert', 'Fortin', 'Fisk', 'Silver', 'Fiore', 'Chadwick', 'Craddock', 'Carrillo', 'Hawkins', 'Darnell', 'Connolly', 'Slaughter', 'Judd', 'Duke', 'Fickett', 'Stearns', 'Dooley', 'Kellogg', 'Strickland', 'Napier', 'Fulton', 'Van', 'Milligan', 'Sutton', 'Franco', 'Wenger', 'MacLeod', 'Colvin', 'Oneal', 'Valentine', 'Hess', 'Porter', 'Barton', 'Weiss', 'Snowden', 'Ceja', 'Huff', 'Lyles', 'Schmidt', 'Rowe', 'Rojas', 'Dailey', 'Culbertson', 'Creamer', 'Rasmussen', 'Warren', 'Milam', 'McLaughlin', 'Pryor', 'Herrmann', 'Bryan', 'Nix', 'Reeder', 'Singleton', 'Doherty', 'Novotny', 'Almond', 'Dorsey', 'Hooker', 'Neuman', 'Linkous', 'Vincent', 'Whatley', 'Monroe', 'Kohl', 'Ferrara', 'McKenzie', 'Summers', 'Humphrey', 'Workman', 'McCall', 'Goetz', 'Stephens', 'Pemberton', 'Waldo', 'Mcwilliams', 'Duggan', 'Hutchinson', 'Wilkes', 'Sturgill', 'Chen', 'Cooley', 'Foust', 'Guzman', 'Maurer', 'Hein', 'Cramer', 'Mendez', 'Prince', 'Schreiber', 'Hendricks', 'Sears', 'Dalton', 'Hooks', 'Ferrell', 'Beamer', 'Barnes', 'Partridge', 'Satterfield', 'Quinones', 'Upton', 'Juarez', 'Maddux', 'Fields', 'Lackey', 'Swartz', 'Paulson', 'Woo', 'Coe', 'Steadman', 'Haney', 'Reno', 'Cortez', 'Peña', 'Culpepper', 'Cheek', 'Cagle', 'Parent', 'Whittington', 'Hightower', 'Daniels', 'Tipton', 'Palmer', 'Garner', 'Randall', 'Hale', 'Wheaton', 'Cyphers', 'Giordano', 'Herrick', 'Schroeder', 'Pauly', 'Lester', 'Ayers', 'Rankin', 'Puckett', 'Lehman', 'Larsen', 'Hegwood', 'Skinner', 'Mott', 'Stump', 'Decker', 'Barr', 'Flynn', 'Means', 'Fischer', 'Wilkerson', 'Salinas', 'Preston', 'Cameron', 'Mims', 'Hoffman', 'Southard', 'Cardenas', 'Voss', 'Hwang', 'Butcher', 'Durst', 'Leech', 'Winstead', 'Cullen', 'Gresham', 'Baca', 'Nestor', 'Stein', 'Adames', 'Lunt', 'Havens', 'Gilman', 'Pickering', 'Leung', 'Zhao', 'Mosley', 'Wingo', 'Glass', 'Ervin', 'Webb', 'Lang', 'Carr', 'Kent', 'Cochran', 'Shelby', 'Haggard', 'Deleon', 'Montoya', 'McIntyre', 'Crowell', 'Wemple', 'Brewster', 'Burnett', 'Lemaster', 'Fajardo', 'Romero', 'Poindexter', 'Beaver', 'Shields', 'Unger', 'Taber', 'Harrington', 'Guthrie', 'Gaudet', 'Pease', 'Dowling', 'Shaw', 'Oconnor', 'Zimmerman', 'Allison', 'Dougherty', 'Clements', 'Plumb', 'McDonald', 'Wilkinson', 'Gardner', 'Calhoun', 'Clemons', 'Stapleton', 'Peoples', 'Holmes', 'Sinclair', 'Thorpe', 'Tanner', 'Keller', 'Pitts', 'Robles', 'Colon', 'Logsdon', 'Rector', 'Quintana', 'Atkins', 'Stark', 'Carroll', 'Sorenson', 'Buckley', 'Daly', 'Zook', 'Busby', 'Potts', 'Stoner', 'Mora', 'Funk', 'English', 'Kuiper', 'Cushing', 'Manson', 'Kern', 'Collier', 'Tolbert', 'MacLean', 'Vargas', 'Burns', 'Lott', 'Caldwell', 'Leavitt', 'Judkins', 'Kuykendall', 'Hildebrand', 'Blount', 'Schaffer', 'Russell', 'Casey', 'Bingham', 'Powell', 'Mueller', 'Watt', 'Stine', 'Heath', 'Beacon', 'Coleman', 'Beltran', 'Hogan', 'Fairbanks', 'Christiansen', 'Riggs', 'Schafer', 'Mccullough', 'Hayes', 'Howland', 'Moody', 'Cartwright', 'Correa', 'Womack', 'Flanagan', 'Lane', 'Alford', 'Brewer', 'Sanborn', 'Wilburn', 'Samuelson', 'Acosta', 'Bearden', 'Stuntz', 'Baumgartner', 'Patten', 'Healy', 'Hahn', 'Lowe', 'Brownlee', 'Bloom', 'Portwood', 'Luciano', 'Foley', 'Dugger', 'Caruso', 'Leon', 'Meek', 'Rucker', 'Whitaker', 'Hurd', 'Zeller', 'Kaufmann', 'Klinger', 'Pardo', 'Strand', 'Leyva', 'Campos', 'Reynoso', 'Doss', 'Andrews', 'Galloway', 'Quincy', 'Canter', 'Hobson', 'Schaub', 'Rapp', 'Wirth', 'Titus', 'Urbina', 'Steele', 'Biddle', 'Berlin', 'Rowland', 'Burch', 'Hodge', 'Bridges', 'Brubaker', 'Funderburk', 'Rainey', 'Savage', 'Wiley', 'Sloan', 'Zavala', 'Pritchett', 'Saunders', 'Pendergrass', 'Brantley', 'Whitman', 'Gallant', 'Borden', 'Lugo', 'Lanier', 'Rutherford', 'Michel', 'Vance', 'Hafner', 'Clay', 'Renfro', 'Rea', 'Mercer', 'Spangler', 'Brock', 'Greer', 'Gipson', 'Rafferty', 'Kasper', 'Felder', 'Wilkins', 'Ladd', 'Nunez', 'Gallo', 'Busch', 'Beckham', 'Carney', 'Goff', 'Spivey', 'Stutzman', 'Washburn', 'Rand', 'Elias', 'Dodds', 'Weldon', 'Page', 'Marquez', 'Chung', 'Maynard', 'Bourgeois', 'Muller', 'Reeves', 'Wallace', 'Zamora', 'Snyder', 'Paterson', 'Tolman', 'Gladden', 'Gould', 'Wickham', 'Schiller', 'Rose', 'Orozco', 'Conway', 'Enriquez', 'Qualls', 'Fogarty', 'Gaither', 'Dumas', 'Flick', 'Overstreet', 'Haugen', 'Hogue', 'Flanders', 'Leatherwood', 'Toth', 'Grigsby', 'Conklin', 'Knox', 'Bemis', 'Faulk', 'Kline', 'Carrasco', 'Hennessey', 'Vazquez', 'Whalen', 'Beach', 'Duran', 'Latta', 'Hebert', 'Hutton', 'Johnston', 'Haas', 'Waldron', 'Bourne', 'Hawley', 'Goss', 'Springer', 'Gannon', 'Fletcher', 'Marler', 'Marshall', 'Spencer', 'Eckert', 'Hancock', 'Grant', 'Guevara', 'Huffman', 'Villarreal', 'Mann', 'Steady', 'Gunn', 'Hurst', 'Ibarra', 'Pinkerton', 'Lofton', 'Mcmahon', 'Langdon', 'Hardwick', 'Wagner', 'Bond', 'Oshea', 'Henson', 'Moreno', 'Waddell', 'Dugan', 'Ott', 'Macdonald', 'Waugh', 'Melton', 'Rosales', 'Barone', 'Kerr', 'Roach', 'Hart', 'McClelland', 'Kemp', 'Silvia', 'Marr', 'Dessert', 'Arroyo', 'Watkins', 'Copeland', 'Grier', 'Baldwin', 'Tucker', 'Lytle', 'Mcgraw', 'Amos', 'Merchant', 'Gilliam', 'Montanez', 'Stallworth', 'Lipscomb', 'Nunn', 'Trammell', 'Larue', 'Rao', 'Persaud', 'Carmichael', 'Mcnair', 'Gillette', 'Shaffer', 'Harrison', 'Beard', 'Johns', 'Severson', 'Connelly', 'Portillo', 'Coyle', 'Rudd', 'Dabney', 'Ingham', 'Knowlton', 'Dodson', 'Cass', 'Forrest', 'Ensign', 'Albertson', 'Glenn', 'Shea', 'Knoll', 'Batchelor', 'Munoz', 'Orr', 'Richards', 'Kinsey', 'Wilhelm', 'Combs', 'Ayala', 'Wendt', 'Avila', 'Tran', 'Bernal', 'Barnard', 'Donohue', 'Wolf', 'Cole', 'Baughman', 'Parnell', 'Jefferson', 'Tenney', 'Horner', 'Lassiter', 'Booker', 'Navarro', 'Hobbs', 'Edge', 'Bullock', 'Clarke', 'McNeal', 'Stone', 'Pinckney', 'Hickman', 'Rendon', 'Noel', 'Bryant', 'Whiting', 'Ledbetter', 'Layne', 'Cormier', 'Bartels', 'Stroud', 'Pinkston', 'Flaherty', 'Perdue', 'Dillard', 'Middleton', 'Woodard', 'Hoyt', 'Dowdy', 'Parrish', 'Obrien', 'Leary', 'Bayes', 'Varner', 'Ricketts', 'Goldsmith', 'Tillery', 'Connors', 'Loomis', 'Stover', 'Small', 'Boyer', 'Booth', 'Bergstrom', 'Borders', 'Kaplan', 'Winn', 'Aguirre', 'Pacheco', 'Baird', 'Serrano', 'Ripley', 'Dempsey', 'Pierce', 'Bunnell', 'Thorne', 'Weeden', 'Wagstaff', 'Crosby', 'Billings', 'Gass', 'Hardesty', 'Weller', 'Hamilton', 'Gillen', 'Thiessen', 'Hensley', 'Burgin', 'Garrison', 'Duffy', 'Elrod', 'Wren', 'Valencia', 'Emanuel', 'Bell', 'Ali', 'Straub', 'Kimble', 'Henderson', 'Hutson', 'Reiter', 'Swan', 'Wheatley', 'Forman', 'Scherer', 'Friesen', 'Kollman', 'Haynes', 'Mills', 'Lister', 'Macon', 'Brinkley', 'Arnold', 'Nash', 'Ruff', 'Holliday', 'Bass', 'Washington', 'Elliott', 'Mahaffey', 'Velazquez', 'Mathis', 'Godfrey', 'Sullivan', 'Nolan', 'Carnahan', 'Cummings', 'Christian', 'Bateman', 'Zahn', 'Weaver', 'Ware', 'Tackett', 'Aldrich', 'Bowers', 'Bales', 'Kendrick', 'Ricker', 'Carpenter', 'Escobedo', 'Haley', 'Faust', 'Hoff', 'Shuler', 'Merritt', 'Ford', 'Underwood', 'Whitehead', 'Barrera', 'Steiner', 'Ellington', 'Dunn', 'Pittman', 'Farrell', 'Crowley', 'Culver', 'Graham', 'Ray', 'Andrade', 'Guerra', 'Albright', 'Bollinger', 'Garrard', 'Hagen', 'Kang', 'Wheeler', 'Starks', 'Dockery', 'Pugh', 'Prewitt', 'Hackney', 'Declercq', 'Lucero', 'Kraus', 'Berrios', 'Thurston', 'Tuck', 'York', 'Tirado', 'Austin', 'Goldberg', 'Camacho', 'Oliva', 'Creech', 'Reynolds', 'Roth', 'Cyr', 'Yates', 'Irwin', 'Fulmer', 'Karl', 'Donaldson', 'Winters', 'Pritchard', 'Lockwood', 'Putnam', 'Rich', 'McGuire', 'Thurman', 'Chance', 'Shelton', 'Beal', 'Farris', 'Sheppard', 'Woodward', 'Alderman', 'Chan', 'Harwood', 'Blankenship', 'Strong', 'Conover', 'Gonzales', 'Livingston', 'Wilder', 'Dyer', 'Hargrove', 'Andersen', 'Ness', 'Sykes', 'Rios', 'Marsh', 'Oshaughnessy', 'Sharpe', 'Warden', 'Cornelius', 'Starling', 'Santoro', 'Murillo', 'Mead', 'Fuentes', 'Simpson', 'Renner', 'Wiggins', 'Montague', 'Truong', 'Bolton', 'Wall', 'Boone', 'Weston', 'Burt', 'Ventura', 'Huerta', 'Crocker', 'Gilmore', 'Rubio', 'Archuleta', 'Malone', 'Pond', 'Giles', 'Sargent', 'Ornelas', 'Biggs', 'Hartman', 'Hendrickson', 'Norwood', 'Odom', 'Holley', 'Rollins', 'Boothe', 'Beauchamp', 'Larkin', 'Lindberg', 'Espinal', 'Goode', 'Barron', 'Gavin', 'Estep', 'Curran', 'Alfaro', 'Silva', 'Ryan', 'Vu', 'Meade', 'Gibson', 'Gentry', 'Pearson', 'Nixon', 'Cardona', 'Hulsey', 'Anthony', 'Burks', 'Downing', 'Weiner', 'Murray', 'Nugent', 'Berkowitz', 'Keating', 'Payne', 'Goldstein', 'Higdon', 'Rice', 'Reilly', 'Arredondo', 'Ferguson', 'Olsen', 'Jacobsen', 'Cline', 'Choi', 'Dillon', 'Gates', 'Willoughby', 'Carrington', 'Mainor', 'Nava', 'Roberson', 'Bauman', 'Gist', 'Lawson', 'Geary', 'Carey', 'Medeiros', 'Kaufman', 'Minton', 'Dahlin', 'Kirby', 'West', 'Hunter', 'Bravo', 'Mooney', 'Jolly', 'Huber', 'Guajardo', 'Trowbridge', 'Organ', 'Fitch', 'Richter', 'Lange', 'Hopper', 'Westfall', 'Ellis', 'Rigney', 'Moser', 'Trujillo', 'Vogt', 'Morrison', 'Marks', 'Rushing', 'Metcalf', 'Landrum', 'Huynh', 'Sawyer', 'Devries', 'Prater', 'Gallagher', 'Mcmullen', 'Olson', 'Contreras', 'Bilodeau', 'Adair', 'Hilliard', 'Thorn', 'Levin', 'Trotter', 'Dunbar', 'Boucher', 'Wakefield', 'Elam', 'Corwin', 'Escalante', 'Deutsch', 'Hester', 'Schmitt', 'Wilt', 'Fong']
        last_names_post = last_names_post[:1000]
    else:
        ValueError(f'Universe {name_universe} is not defined, please provide additional names in `parameters_entity_fill_func` function for replacing the $entity_X with names')

    if any(' ' in last_name for last_name in last_names_post):
        raise ValueError('last names post should not contain space')
    parameters_universe_post = {'first_names_post': first_names_post, 'last_names_post': last_names_post}
    return parameters_universe_post

## Post-processing step 1: cut_paragraph_wo_numbers to get intermediate samples `generated_paragraph2`
def cut_paragraph_wo_numbers_func(paragraphs):
    cut_paragraphs = cut_paragraph_func(paragraphs)
    cut_paragraph_wo_numbers = [remove_initial_numbering(p) for p in cut_paragraphs]
    return cut_paragraph_wo_numbers

def get_intermediate_sample(e, prompt_parameters, model_parameters, data_folder, rechecking=True):
    """
    Extract single_ei which is the last iteration.
    If the generated_paragraph is still not working, the generated paragraph is set to None and has_passed=False
    In the generated_paragraph, the external entities are still $entity_X and have not been replaced yet
    """
    #_, has_verif_vector = iterative_generate_paragraphs_func(prompt_parameters, model_parameters, itermax, data_folder, env_file, False)
    output = get_single_ei(e, "last", prompt_parameters, model_parameters, data_folder, rechecking)
    event, meta_event, generated_paragraph, has_passed_verif_direct, has_passed_verif_llm, _, _, _, iteration = output
    has_passed = has_passed_verif_direct and has_passed_verif_llm

    # apply initial post processing
    if has_passed:
        cut_generated_paragraphs = cut_paragraph_wo_numbers_func(generated_paragraph)
        generated_paragraph2 = '\n\n'.join(cut_generated_paragraphs)
    else:
        generated_paragraph2 = None

    return generated_paragraph2, has_passed, e, iteration, event, meta_event

## Post-processing step 2: replace $entity_X from `generated_paragraph2` to get `final_generated_paragraph`
def count_unique_entities(str):
    # Find all occurrences of $entity_X in the string
    pattern = r'\$entity_(\d+)'
    matches = re.findall(pattern, str)
    
    # Convert matches to integers and get unique values
    unique_entities = set(map(int, matches))
    
    # Return the count of unique entities
    return len(unique_entities)

def extract_entities(entities_count_per_e, post_entities):
    entities = []
    index = 0
    for count in entities_count_per_e:
        if count == 0:
            entities.append([])
        else:
            entities.append(post_entities[index:index+count])
            index += count
    return entities

def replace_entities(str, replacements):
    def replace_match(match):
        index = int(match.group(1)) - 1
        return replacements[index] if index < len(replacements) else match.group(0)
    
    pattern = r'\$entity_(\d+)'
    return re.sub(pattern, replace_match, str)

def replace_dollar_entities(all_generated_paragraphs2, post_entities):
    """
    Usage example:

    Input:
        str = 'The crisp autumn air carried $entity_1 a hint of foreboding $entity_2 dsdd fdfds'
        str2 = 'The crisp autumn air carried a hint of foreboding dsdd fdfds'
        str3 = None
        str4 = 'The crisp $entity_1 autumn air carried $entity_2 a hint of foreboding $entity_3 dsdd fdfds'
        all_generated_paragraphs2 = [str, str2, str3, str4]
        post_entities = ['Jean Taylor', 'May Obama', 'Jean Jean', 'Mike Bond', 'Helena Jordan', 'Extra Person']

        replace_dollar_entities(all_generated_paragraphs2, post_entities)

    Output:
        (['The crisp autumn air carried Jean Taylor a hint of foreboding May Obama dsdd fdfds',
        'The crisp autumn air carried a hint of foreboding dsdd fdfds',
        None,
        'The crisp Jean Jean autumn air carried Mike Bond a hint of foreboding Helena Jordan dsdd fdfds'],
        [['Jean Taylor', 'May Obama'],
        [],
        [],
        ['Jean Jean', 'Mike Bond', 'Helena Jordan']])
    """
    entities_count_per_e = [count_unique_entities(str) if str is not None else 0 for str in all_generated_paragraphs2]
    if sum(entities_count_per_e) > len(post_entities):
        raise ValueError('More post_entities are needed to fill all the $entity of all the generated paragraphs')
    
    # list of post entities necessary for each index e (following the order of post_entities)
    used_post_entities_per_e = extract_entities(entities_count_per_e, post_entities)

    # replace the entities
    final_generated_paragraphs = [replace_entities(p, post_ent) if p is not None else None for (p, post_ent) in zip(all_generated_paragraphs2, used_post_entities_per_e)]
    return final_generated_paragraphs, used_post_entities_per_e

## Adding number of tokens
def count_tokens(text):
    """# Example usage
    sample_text = "This is a sample sentence for Claude 3.5 Sonnet token counting."
    token_count = count_tokens(sample_text)
    print(token_count)
    """
    # Use the cl100k_base encoding, which is close to what Claude might use
    encoding = get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    return len(tokens)

## Final function
def get_final_samples(prompt_parameters, model_parameters, data_folder, rechecking=True):
    # List of (generated_paragraph2, has_passed, e, iteration, event, meta_event)
    intermediate_samples = [get_intermediate_sample(e, prompt_parameters, model_parameters, data_folder, rechecking) for e in range(prompt_parameters['nb_events'])]

    all_generated_paragraph2 = [x[0] for x in intermediate_samples]
    all_has_passed = [x[1] for x in intermediate_samples]
    all_e = [x[2] for x in intermediate_samples]
    all_iteration = [x[3] for x in intermediate_samples]
    all_event = [x[4] for x in intermediate_samples]
    all_meta_event = [x[5] for x in intermediate_samples]

    post_entities = generate_post_universe(nb_names = 100000, name_universe = prompt_parameters['name_universe'], seed_universe = 0)
    final_generated_paragraphs, used_post_entities_per_e = replace_dollar_entities(all_generated_paragraph2, post_entities)

    nb_tokens = [count_tokens(text) if text is not None else None for text in final_generated_paragraphs]

    final = {e: {'paragraphs': p, 'is_valid': h, 'nb_tokens': tok, 'event_idx': e, 'iter_idx': i, 'event': ev, 'meta_event': m, 'post_entities': post} for 
             (p, h, tok, e, i, ev, m, post) in zip(final_generated_paragraphs, all_has_passed, nb_tokens, all_e, all_iteration, all_event, all_meta_event, used_post_entities_per_e)}
    return final
