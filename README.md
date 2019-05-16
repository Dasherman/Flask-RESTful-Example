Adhv https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166

Dit is een manier om een model in een API te verwerken, zodat deze opgeroepen kan worden via web requests (bijvoorbeeld via de requests module in Python).

Het begint bij een model. Maak een class voor je model. Maak vervolgens een instance van deze class en train hem.
Dan kan je dit object opslaan met pickle.dump(). Dit zet je getrainde model om in een bestandje.

Nu kan je beginnen aan de API zelf. Hiervoor heb je Flask nodig en Flask-RESTful. Importeer ook de class die je net hebt gemaakt voor je model en laad het model in met pickle.load().
Het begin van het API-script is standaard.  Daarna moet je een nieuwe class maken met een get-functie erin. Deze functie verwerkt de binnenkomende requests en moet een dictionary returnen waarbij keys én values strings zijn (JSON).

Dan maak je het script af met standaarddingen.

Als je het script nu runt kun je requests doen (zie bijvoorbeeld 'request test.ipynb').

PS zie ook https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b865c166 voor een manier om meerdere modellen in de API te stoppen.