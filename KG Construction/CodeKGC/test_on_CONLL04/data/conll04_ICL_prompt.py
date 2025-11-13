""" Newspaper ` Explains ' U.S. Interests Section Events FL1402001894 Havana Radio Reloj Network in Spanish 2100 GMT 13 Feb 94 """
extract = Extract([Triple(organization('Radio Reloj Network'), OrgBased_In(), location('Havana')),])
""" ` ` If it does not snow , and a lot , within this month we will have no water to submerge 150 , 000 hectares ( 370 , 500 acres ) of rice , ' ' said Bruno Pusterla , a top official of the Italian Agricultural Confederation . """
extract = Extract([Triple(person('Bruno Pusterla'), Work_For(), organization('Italian Agricultural Confederation')),])
""" The self-propelled rig Avco 5 was headed to shore with 14 people aboard early Monday when it capsized about 20 miles off the Louisiana coast , near Morgan City , Lifa said. """
extract = Extract([Triple(location('Morgan City'), Located_In(), location('Louisiana')),])
""" Annie Oakley , also known as Little Miss Sure Shot , was born Phoebe Ann Moses in Willowdell , Darke County , in 1860 . """
extract = Extract([Triple(person('Annie Oakley'), Live_In(), location('Willowdell , Darke County')), Triple(person('Little Miss Sure Shot'), Live_In(), location('Willowdell , Darke County')), Triple(person('Phoebe Ann Moses'), Live_In(), location('Willowdell , Darke County')),])
""" The viewers of \" JFK \" and \" The Men Who Killed Kennedy \" never learn about these facts , nor do they ever learn about all of the other massive body of evidence that conclusively proves beyond a reasonable doubt that Oswald was the lone gunman who killed President Kennedy and Officer Tippit and that there was no coverup by Earl Warren or by the Warren Commission. ; """
extract = Extract([Triple(person('Oswald'), Kill(), person('President Kennedy')), Triple(person('Oswald'), Kill(), person('Officer Tippit')),])
""" The sky is blue and the grass is green. """
extract = Extract([])