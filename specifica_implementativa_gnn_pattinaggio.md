# Specifica implementativa

## Siamese GATv2 per la valutazione della qualità degli elementi nel pattinaggio di figura

**Versione:** 1.0\
**Data:** 11 novembre 2026\
**Tipo di progetto:** MVP accademico --- Big Data and Deep Learning

------------------------------------------------------------------------

## 1. Scopo e obiettivi

Il progetto mira a realizzare un sistema di *Action Quality Assessment*
(AQA) per il pattinaggio di figura. Il sistema riceve lo scheletro
tridimensionale di un atleta durante un elemento tecnico e stima quanto
la posa osservata sia simile a esecuzioni di riferimento di qualità
elevata.

L'approccio scelto è una rete neurale a grafo Siamese basata su
**GATv2**, addestrata con **Triplet Loss**. La rete apprende una
rappresentazione latente dello scheletro; la distanza tra gli embedding
dell'esecuzione osservata e di un riferimento costituisce la misura
primaria di dissimilarità.

Il progetto è un MVP: deve dimostrare la fattibilità della pipeline
end-to-end su un insieme ristretto di elementi del pattinaggio, non
realizzare un sistema di giudizio ufficiale utilizzabile in gara.

### 1.1 Obiettivi specifici

1.  Recuperare e analizzare dataset di pattinaggio con scheletri
    pre-estratti e annotazioni di qualità.
2.  Definire un formato comune per gli scheletri e per le relative
    annotazioni.
3.  Selezionare un piccolo insieme di elementi tecnici sufficientemente
    rappresentati.
4.  Canonicalizzare le coordinate 3D per ridurre le differenze dovute a
    traslazione, scala e orientamento.
5.  Convertire ogni posa in un grafo con topologia e semantica dei nodi
    coerenti.
6.  Addestrare una Siamese GATv2 con triplette di grafi.
7.  Valutare se la distanza appresa è coerente con la qualità annotata.
8.  Visualizzare i risultati e, ove possibile, i pesi di attenzione
    della rete.

### 1.2 Fuori ambito

Non rientrano nell'MVP:

-   valutazione completa di un programma di pattinaggio;
-   riconoscimento di tutti gli elementi tecnici;
-   ricostruzione 3D da video multi-camera;
-   generazione automatica di un punteggio ufficiale ISU;
-   inferenza real-time;
-   addestramento di un nuovo pose estimator;
-   generalizzazione dimostrata a tutte le telecamere, condizioni di
    ripresa e categorie di atleti.

------------------------------------------------------------------------

## 2. Decisioni progettuali consolidate

  -----------------------------------------------------------------------
  Aspetto                             Decisione
  ----------------------------------- -----------------------------------
  Sport                               Pattinaggio di figura

  Dataset principale                  FineFS

  Dataset integrativo                 MMFS, previa verifica di
                                      compatibilità

  Rappresentazione                    Scheletro 3D a 17 joint

  Pose estimator dei dati accademici  HRNet, secondo la descrizione dei
                                      dataset

  Topologia                           Grafo canonico coerente con i 17
                                      joint

  Modello                             Siamese GATv2

  Feature degli archi                 Sì, tramite `edge_attr`

  Obiettivo di training               Metric learning

  Loss                                Triplet Margin Loss con margine
                                      adattivo

  Segnale di qualità                  GOE, quando disponibile e
                                      associabile al singolo elemento

  Output primario                     Distanza tra embedding

  Explainability                      Pesi di attenzione GATv2

  Hardware                            Portatile personale con GPU NVIDIA;
                                      modello esatto e VRAM da verificare

  Durata prevista                     14 giorni
  -----------------------------------------------------------------------

Le decisioni costituiscono la baseline. Le sezioni successive
specificano come verificarle e implementarle.

------------------------------------------------------------------------

## 3. Dataset e gestione dei dati

### 3.1 FineFS --- dataset principale

**Riferimento:** Fine-grained Figure Skating Dataset (FineFS),
repository indicato nella revisione:\
https://github.com/yanliji/FineFS-dataset

Secondo la documentazione esaminata nella revisione, FineFS contiene:

-   1.167 esibizioni;
-   sotto-elementi tecnici segmentati con intervalli temporali;
-   punteggi GOE (*Grade of Execution*) associati agli elementi;
-   scheletri 3D a 17 joint, distribuiti in file `.npz`;
-   una raccolta di scheletri indicata nella revisione come circa 828
    MB.

FineFS è il dataset principale perché associa, almeno secondo la
descrizione disponibile, elementi segmentati, scheletri e valutazioni di
qualità a livello di elemento.

**Verifiche obbligatorie prima del training:**

1.  Verificare che repository, link di download e file siano ancora
    accessibili.
2.  Ispezionare il contenuto di alcuni `.npz`, senza assumere a priori
    nomi o shape degli array.
3.  Verificare dimensioni, unità, sistema di riferimento e significato
    degli assi delle coordinate.
4.  Verificare se le coordinate sono effettivamente 3D e quale procedura
    di stima le ha prodotte.
5.  Ricostruire la relazione tra identificativo dell'esibizione,
    intervallo temporale, elemento tecnico, scheletro e GOE.
6.  Verificare se il GOE è associato al singolo sotto-elemento o
    soltanto a un livello più aggregato.
7.  Controllare valori mancanti, keypoint non rilevati, duplicati e
    segmenti vuoti.

Non si deve iniziare il training finché la corrispondenza tra scheletro,
elemento e annotazione non è stata verificata.

### 3.2 MMFS --- dataset integrativo

**Riferimento:** repository MMFS indicato nella revisione:\
https://github.com/dingyn-Reno/MMFS

MMFS viene considerato una possibile fonte aggiuntiva di scheletri e
annotazioni, inclusi punteggi GOE e Base Value secondo la descrizione
riportata nella revisione.

L'integrazione non è automatica. Prima di unire i dati occorre
confrontare:

-   convenzione e ordine dei 17 joint;
-   dimensionalità e sistema di coordinate;
-   unità di misura e convenzioni degli assi;
-   modalità di estrazione dello scheletro;
-   segmentazione temporale degli elementi;
-   definizione e scala del GOE;
-   identificativi degli atleti e delle esibizioni;
-   eventuali sovrapposizioni tra FineFS e MMFS.

Se queste informazioni non sono compatibili o non sono ricostruibili,
MMFS deve essere mantenuto separato oppure escluso dall'MVP. Non si
devono concatenare i campioni soltanto perché entrambi contengono 17
keypoint.

### 3.3 Selezione degli elementi tecnici

L'MVP deve concentrarsi su pochi elementi, preferibilmente pose
mantenute o elementi con una fase riconoscibile e relativamente stabile.

Tra gli esempi da valutare:

-   Camel Spin;
-   Sit Spin;
-   Layback Spin.

La scelta definitiva deve dipendere da:

1.  numero di campioni disponibili;
2.  qualità e completezza degli scheletri;
3.  disponibilità del GOE per il singolo elemento;
4.  consistenza della segmentazione;
5.  comparabilità biomeccanica tra le esecuzioni.

La selezione deve essere effettuata con un'analisi esplorativa del
dataset. Non si deve presumere che tutti gli elementi elencati siano
presenti in quantità sufficiente.

### 3.4 Schema dati unificato

Ogni campione deve essere rappresentato da un record con almeno i
seguenti campi logici:

  Campo                   Descrizione
  ----------------------- -------------------------------------------------
  `sample_id`             Identificativo univoco del campione
  `dataset_id`            Dataset di provenienza
  `performance_id`        Identificativo dell'esibizione
  `athlete_id`            Identificativo atleta, se disponibile
  `element_label`         Classe dell'elemento tecnico
  `start_frame`           Frame iniziale del segmento, se disponibile
  `end_frame`             Frame finale del segmento, se disponibile
  `keypoints`             Coordinate 3D dei 17 joint
  `keypoint_confidence`   Confidenza per joint, se disponibile
  `goe`                   GOE associato all'elemento, se disponibile
  `source_metadata`       Metadati originali necessari alla tracciabilità

Il formato interno deve preservare i metadati originali. La
trasformazione nel formato comune non deve eliminare la possibilità di
risalire al campione sorgente.

### 3.5 Split dei dati

La suddivisione training/validation/test deve avvenire **per esibizione
e, quando possibile, per atleta**, non per singolo frame.

In questo modo si riduce il rischio che frame quasi identici della
stessa esecuzione compaiano in split diversi.

Tutti i campioni derivati dalla stessa esecuzione, inclusi quelli
generati tramite augmentation, devono rimanere nello stesso split del
campione originale.

Le percentuali degli split vanno definite dopo aver misurato la
numerosità effettiva per classe. Come punto di partenza si può usare 70%
training, 15% validation e 15% test, modificandole se la dimensione del
dataset non consente una ripartizione stabile.

------------------------------------------------------------------------

## 4. Rappresentazione temporale della posa

Il progetto originario prevede il confronto di una media di pochi frame
per rappresentare una posa o una fase dell'elemento.

Per ogni segmento selezionato:

1.  identificare la finestra temporale corrispondente alla posa o alla
    fase di interesse;
2.  selezionare un numero limitato di frame;
3.  verificare la qualità dei keypoint in ciascun frame;
4.  canonicalizzare ogni scheletro;
5.  aggregare le coordinate o le feature dei frame validi.

La baseline è la media delle coordinate canonicalizzate dei frame
selezionati:

\[ `\bar`{=tex}{P} = `\frac{1}{T}`{=tex}`\sum`{=tex}\_{t=1}\^{T}P_t \]

dove (P_t `\in `{=tex}`\mathbb{R}`{=tex}\^{17 `\times 3`{=tex}}) è lo
scheletro canonicalizzato al frame (t).

**Attenzione:** la media delle coordinate è appropriata soltanto quando
i frame rappresentano la stessa fase e sono sufficientemente allineati.
Non si devono mediare frame appartenenti a fasi cinematiche diverse,
perché il risultato potrebbe non corrispondere a una posa realmente
assunta dall'atleta.

Per l'MVP si deve usare una finestra temporale coerente con la
segmentazione disponibile. La dimensione esatta della finestra e il
numero di frame sono iperparametri da verificare empiricamente.

------------------------------------------------------------------------

## 5. Canonicalizzazione geometrica 3D

### 5.1 Motivazione

Le coordinate grezze dipendono da traslazione, scala e orientamento del
sistema di riferimento. Due pose biomeccanicamente simili potrebbero
quindi apparire diverse alla rete.

Prima della costruzione del grafo si applicano tre trasformazioni:

1.  root-centering;
2.  scale normalization;
3.  yaw alignment.

Queste trasformazioni riducono alcune differenze geometriche, ma non
garantiscono invarianza completa al punto di vista. In particolare,
l'allineamento yaw non corregge automaticamente differenze di pitch,
roll, prospettiva o errori di ricostruzione 3D.

### 5.2 Root-centering

Si definisce la root come punto medio delle anche, assumendo la
convenzione COCO a 17 joint e verificando gli indici rispetto al file
sorgente:

\[ p\_{`\text{pelvis}`{=tex}} = `\frac{p_{11}+p_{12}}{2}`{=tex} \]

Per ogni joint:

\[ p'*i = p_i - p*{`\text{pelvis}`{=tex}} \]

La trasformazione rimuove la traslazione globale dello scheletro.

### 5.3 Scale normalization

Si definisce il punto medio delle spalle:

\[ p\_{`\text{shoulders}`{=tex}} = `\frac{p_5+p_6}{2}`{=tex} \]

La scala di riferimento è la lunghezza del busto:

\[ s = \|p\_{`\text{shoulders}`{=tex}}-p\_{`\text{pelvis}`{=tex}}\|\_2
\]

Le coordinate vengono normalizzate:

\[ p''\_i = `\frac{p'_i}{s}`{=tex} \]

Se (s) è nullo, troppo piccolo o non valido, il campione deve essere
marcato come non valido oppure gestito con una strategia esplicita. Non
si deve dividere per zero né sostituire silenziosamente valori mancanti.

La normalizzazione riduce le differenze di scala e corporatura, ma può
anche rimuovere informazioni sulle proporzioni corporee. Questa scelta
va documentata come assunzione del modello.

### 5.4 Yaw alignment

Si utilizza il vettore tra le anche:

\[ v\_{`\text{hip}`{=tex}} = p\_{12}-p\_{11} \]

Sul piano X-Y si calcola:

\[ `\theta `{=tex}= `\operatorname{atan2}`{=tex}(v_y,v_x) \]

Si applica una rotazione inversa attorno all'asse Z:

\[ R_z(-`\theta`{=tex})=
```{=tex}
\begin{bmatrix}
\cos(-\theta)&-\sin(-\theta)&0\\
\sin(-\theta)&\cos(-\theta)&0\\
0&0&1
\end{bmatrix}
```
\]

Le coordinate canonicalizzate sono:

\[ p'''\_i = R_z(-`\theta`{=tex})p''\_i \]

L'implementazione deve verificare che gli assi X, Y e Z dei dataset
abbiano il significato assunto da questa formula. Se la convenzione
degli assi è diversa, la rotazione va adattata.

### 5.5 Controlli e limiti

La pipeline deve registrare:

-   lunghezza del busto utilizzata per la scala;
-   angolo di yaw stimato;
-   presenza di joint mancanti o non validi;
-   eventuali scheletri scartati;
-   distribuzione delle coordinate prima e dopo la canonicalizzazione.

Va inoltre verificato il comportamento in caso di scheletro specchiato.
L'allineamento del vettore delle anche non risolve da solo l'ambiguità
destra/sinistra. Non si deve applicare un flip automatico senza una
regola verificata e coerente con le etichette.

La robustezza rispetto a viste differenti deve essere valutata
sperimentalmente su dati disponibili. Non è necessario disporre della
stessa esecuzione ripresa da due telecamere, ma l'assenza di coppie
multi-view impedisce di dimostrare direttamente l'invarianza a viste
diverse.

------------------------------------------------------------------------

## 6. Costruzione del grafo

### 6.1 Topologia

Ogni posa viene rappresentata come un grafo:

\[ G=(V,E,X,E_f) \]

dove:

-   (V): 17 nodi corrispondenti ai joint;
-   (E): connessioni anatomiche dello scheletro;
-   (X): feature dei nodi;
-   (E_f): feature degli archi.

La topologia deve essere fissa e coerente con lo standard dei 17
keypoint adottato dai dataset. La lista esatta delle connessioni deve
essere definita una volta, verificata rispetto alla convenzione
HRNet/COCO utilizzata e salvata come configurazione del progetto.

Non si devono assumere equivalenti topologie che differiscono per ordine
o semantica dei nodi.

### 6.2 Node features

Le feature iniziali dei nodi devono descrivere la geometria
canonicalizzata. Baseline:

\[ x_i = \[x_i,y_i,z_i\] \]

È possibile aggiungere feature derivate, come vettori direzionali delle
ossa o indicatori di validità/confidenza, se disponibili e verificati.

Per un osso tra i nodi (i) e (j):

\[ u\_{ij}=`\frac{p_j-p_i}{\|p_j-p_i\|_2+\epsilon}`{=tex} \]

Le feature aggiuntive devono essere introdotte in modo controllato,
confrontando la baseline con la variante estesa.

### 6.3 Edge features

Gli archi devono contenere informazioni geometriche, così che GATv2 non
consideri soltanto la connettività.

Feature candidate:

-   vettore direzionale normalizzato dell'osso;
-   lunghezza dell'osso dopo normalizzazione;
-   coseno dell'angolo articolare associato, quando definibile.

Per tre joint (i,j,k), il coseno dell'angolo in (j) è:

\[ `\cos`{=tex}(`\theta`{=tex}\_{ijk}) =
`\frac{(p_i-p_j)\cdot(p_k-p_j)}`{=tex}
{\|p_i-p_j\|\_2\|p_k-p_j\|\_2+`\epsilon`{=tex}} \]

La rappresentazione finale di `edge_attr` deve avere dimensione costante
per tutti gli archi e per tutti i campioni.

**Nota implementativa:** se una feature è definita a livello di
articolazione e non di singolo arco, occorre stabilire una regola
deterministica per associarla agli archi incidenti. Tale regola va
documentata e mantenuta uguale in training e inference.

### 6.4 Formato PyTorch Geometric

Ogni grafo deve essere serializzabile come oggetto
`torch_geometric.data.Data`, con almeno:

-   `x`: feature dei nodi;
-   `edge_index`: connettività;
-   `edge_attr`: feature degli archi;
-   `y` o metadati separati: annotazioni utilizzate dal sampler e dalla
    valutazione;
-   identificativo del campione e del dataset conservati nei metadati o
    in un indice esterno.

La serializzazione deve consentire di ricostruire il campione originale.

------------------------------------------------------------------------

## 7. Architettura del modello

### 7.1 Siamese GATv2

La rete è composta da due rami con pesi condivisi. Ogni ramo riceve un
grafo e produce un embedding nello stesso spazio latente.

Schema:

``` text
Grafo A ──> Encoder GATv2 condiviso ──> Embedding zA ──┐
                                                       ├─> Distanza latente
Grafo B ──> Encoder GATv2 condiviso ──> Embedding zB ──┘
```

Per il training triplet:

``` text
Anchor   ──> Encoder condiviso ──> zA
Positive ──> Encoder condiviso ──> zP
Negative ──> Encoder condiviso ──> zN
```

### 7.2 Scelta di GATv2Conv

Si utilizza `GATv2Conv` di PyTorch Geometric anziché `GATConv` standard.

Motivazione progettuale: GATv2 consente un meccanismo di attenzione
dinamica, in cui il peso assegnato ai vicini dipende in modo più
flessibile dal nodo interrogante e dalle feature coinvolte.

La configurazione deve includere `edge_dim` per consentire al layer di
utilizzare le feature degli archi:

``` python
GATv2Conv(
    in_channels=...,
    out_channels=...,
    heads=...,
    edge_dim=...
)
```

### 7.3 Encoder

L'encoder deve comprendere:

1.  uno o più layer GATv2;
2.  funzione di attivazione;
3.  eventuale dropout;
4.  pooling globale per ottenere una rappresentazione del grafo;
5.  layer lineare di proiezione nello spazio latente.

Schema concettuale:

``` text
Node features + Edge features
          |
       GATv2
          |
       GATv2
          |
  Global Mean Pooling
          |
   Linear projection
          |
      Embedding z
```

Dimensioni iniziali da sperimentare:

-   embedding: 32 o 64;
-   attention heads: 4 come configurazione iniziale;
-   dropout: da selezionare tramite validation;
-   numero di layer: limitato per ridurre il rischio di overfitting.

Questi valori sono iperparametri iniziali, non risultati già validati.
La dimensione degli input e degli output dei layer deve essere coerente
con il numero di head e con l'eventuale concatenazione delle head.

### 7.4 Pesi condivisi

I tre input della tripla devono attraversare la stessa istanza
dell'encoder. Non si devono creare encoder indipendenti con parametri
separati.

### 7.5 Regolarizzazione

Dato il numero potenzialmente limitato di campioni reali:

-   mantenere l'encoder compatto;
-   usare dropout;
-   applicare early stopping sulla validation;
-   monitorare la distanza tra embedding;
-   evitare augmentation prima dello split;
-   controllare la sovrapposizione tra triplette;
-   confrontare training e validation loss per individuare overfitting.

------------------------------------------------------------------------

## 8. Training: Triplet Loss e costruzione delle triplette

### 8.1 Motivazione

Non si assume che esistano etichette precise di penalità per ogni
singola posa. Il GOE, quando associato al singolo elemento, viene
utilizzato per costruire relazioni di qualità tra campioni.

La Triplet Loss permette di apprendere una metrica senza dover conoscere
una penalità numerica esatta per ogni posa.

### 8.2 Definizione delle triplette

Ogni tripla contiene:

-   **Anchor (A):** un'esecuzione di un determinato elemento;
-   **Positive (P):** un'esecuzione dello stesso elemento con qualità
    comparabile o superiore;
-   **Negative (N):** un'esecuzione dello stesso elemento con qualità
    inferiore.

Le triplette devono rispettare la stessa classe di elemento, per evitare
che il modello impari a distinguere elementi tecnici diversi invece
della qualità esecutiva.

Una regola iniziale, da verificare sulla distribuzione reale dei GOE, è:

-   positivi: campioni con GOE elevato, ad esempio GOE ≥ +2;
-   negativi: campioni con GOE basso o non positivo, ad esempio GOE ≤ 0.

Queste soglie sono quelle proposte nella revisione e devono essere
adattate alla scala effettiva, alla distribuzione e alla definizione del
GOE nel dataset. Non sono soglie ufficiali di penalità per singola
articolazione.

Quando possibile, l'Anchor deve essere scelto tra i campioni di qualità
elevata e il Positive tra campioni di qualità analoga. La selezione deve
evitare che la stessa esecuzione sia usata come Anchor e Positive.

### 8.3 Triplet Margin Loss

La loss standard è:

\[ L(A,P,N)= `\max`{=tex}`\left`{=tex}( \|f(A)-f(P)\|\_2 -
\|f(A)-f(N)\|\_2 + `\alpha`{=tex}, 0 `\right`{=tex}) \]

dove (f) è l'encoder Siamese e (`\alpha`{=tex}) è il margine.

La rete deve imparare a mantenere:

\[ d(A,P)+`\alpha `{=tex}\< d(A,N) \]

### 8.4 Margine adattivo

Si prevede un margine dipendente dalla severità della differenza di
qualità tra Anchor e Negative.

Valori iniziali proposti nella revisione:

  Gravità del negativo sintetico/annotato     Margine iniziale
  ----------------------------------------- ------------------
  Lieve                                                    0.2
  Medio                                                    0.6
  Grave                                                    1.0

Questi valori sono iperparametri sperimentali. Non rappresentano una
conversione ufficiale tra GOE e distanza latente.

**Punto metodologico importante:** un margine adattivo richiede una
regola riproducibile che associ la coppia Anchor-Negative a un livello
di differenza. Se il GOE non permette di determinare tale livello, si
deve usare una strategia più semplice: margine fisso oppure fasce di
qualità definite sui dati disponibili.

### 8.5 Data augmentation

L'augmentation geometrica è una strategia di supporto, non un sostituto
delle annotazioni reali.

Possibili perturbazioni:

-   variazioni controllate degli angoli articolari;
-   perturbazioni locali di gomiti, ginocchia o anche;
-   rumore gaussiano limitato sulle coordinate;
-   piccole rotazioni e variazioni geometriche compatibili con la posa.

Ogni trasformazione deve preservare la plausibilità anatomica e non
cambiare accidentalmente la classe dell'elemento.

I campioni sintetici devono avere metadati che identifichino il campione
sorgente, il tipo di perturbazione e la sua intensità.

Non si devono attribuire automaticamente penalità ufficiali ISU alle
perturbazioni geometriche. Le soglie angolari e i valori di penalità
menzionati nella revisione devono essere verificati sul regolamento
applicabile e non trattati come label certificate senza tale verifica.

L'augmentation deve essere applicata esclusivamente ai campioni del
training set.

### 8.6 Strategia di sampling

Il sampler deve:

1.  selezionare una classe di elemento;
2.  selezionare un Anchor;
3.  selezionare un Positive della stessa classe e qualità comparabile;
4.  selezionare un Negative della stessa classe ma di qualità inferiore;
5.  assegnare il margine secondo la regola configurata;
6.  evitare campioni appartenenti a split differenti.

In una fase successiva si può introdurre *semi-hard negative mining*,
scegliendo negativi più difficili da distinguere. Per il primo MVP è
preferibile implementare un sampler semplice, verificabile e
riproducibile.

------------------------------------------------------------------------

## 9. Conversione della distanza in indicatore di qualità

La distanza euclidea tra embedding è:

\[ d(A,R)=\|f(A)-f(R)\|\_2 \]

dove (A) è il campione dell'atleta e (R) è un riferimento di qualità
elevata dello stesso elemento.

La distanza è l'output primario del modello. Una distanza minore indica
maggiore somiglianza nello spazio appreso.

Per presentare un indicatore più intuitivo si può applicare una
trasformazione monotona, ad esempio:

\[ q(d)=`\exp`{=tex}(-`\beta `{=tex}d) \]

dove (`\beta`{=tex}) è un parametro da calibrare sulla validation.

Il risultato può essere riportato su scala 0--100:

\[ Q=100`\cdot `{=tex}q(d) \]

Questo valore è un **indice interno di similarità**, non un punteggio
ufficiale, una percentuale di correttezza certificata o una stima
diretta del GOE.

La calibrazione deve essere effettuata soltanto sulla validation. Il
test set deve rimanere separato fino alla valutazione finale.

Se sono disponibili più riferimenti per un elemento, si può confrontare
l'atleta con un prototipo ottenuto aggregando gli embedding dei campioni
di alta qualità, oppure con più riferimenti individuali. Per l'MVP è
sufficiente definire una strategia unica e mantenerla costante durante
la valutazione.

------------------------------------------------------------------------

## 10. Explainability

La componente di explainability utilizza i pesi di attenzione restituiti
da `GATv2Conv`.

Con `return_attention_weights=True` è possibile recuperare i
coefficienti di attenzione associati agli archi.

Obiettivo della visualizzazione:

-   rappresentare lo scheletro sul piano;
-   associare un'intensità visiva ai collegamenti;
-   evidenziare le connessioni a cui il modello assegna maggiore
    attenzione;
-   confrontare visivamente esecuzione e riferimento.

La visualizzazione deve essere interpretata con cautela: un peso di
attenzione elevato indica che il layer assegna maggiore peso a una
connessione nel proprio calcolo, ma non dimostra da solo causalità né
corrispondenza diretta con una penalità regolamentare.

Per una visualizzazione leggibile:

1.  aggregare i pesi tra le head secondo una regola esplicita, ad
    esempio media;
2.  definire una normalizzazione dei valori;
3.  mappare i valori sugli archi dello scheletro;
4.  riportare il nome dell'elemento e la distanza latente;
5.  mantenere la stessa scala cromatica tra campioni confrontati.

La richiesta dei pesi di attenzione va abilitata in fase di
inferenza/analisi, senza appesantire inutilmente il training.

------------------------------------------------------------------------

## 11. Pipeline di inferenza e demo

La demo deve mostrare il funzionamento su pochi video esterni al
training.

Pipeline prevista:

``` text
Video
  |
Pose estimation (MediaPipe o altro estimator selezionato)
  |
Mapping dei keypoint sullo standard canonico a 17 nodi
  |
Controllo qualità dei keypoint
  |
Selezione della finestra temporale
  |
Canonicalizzazione 3D (se sono disponibili coordinate 3D compatibili)
  |
Costruzione del grafo
  |
Siamese GATv2
  |
Embedding e distanza dal riferimento
  |
Indice di similarità + visualizzazione
```

### 11.1 Compatibilità MediaPipe-HRNet

La revisione propone di usare MediaPipe per la demo, proiettando i suoi
33 landmark sui 17 joint dello standard COCO.

Questa operazione richiede una mappatura esplicita e verificata per ogni
joint. Non è sufficiente selezionare indici senza controllare la
semantica dei landmark.

Inoltre, la compatibilità non è soltanto topologica:

-   HRNet e MediaPipe possono produrre coordinate in sistemi di
    riferimento differenti;
-   le coordinate 3D possono avere significati e scale differenti;
-   il bias del pose estimator può essere diverso;
-   la proiezione da 33 a 17 punti non rende automaticamente equivalenti
    le distribuzioni.

**Vincolo:** il modello addestrato su scheletri FineFS/MMFS non deve
ricevere direttamente coordinate MediaPipe senza una fase di verifica
del dominio e della rappresentazione.

Per il primo MVP è accettabile mostrare la pipeline di inferenza su un
campione proveniente dallo stesso dominio dei dati di training. La demo
MediaPipe è un'estensione subordinata alla compatibilità tecnica.

### 11.2 Pose estimator alternativo

La revisione cita MediaPipe come soluzione pratica per la demo, ma
segnala i suoi limiti in contesti sportivi complessi. Cita inoltre
RTMPose e ViTPose come alternative.

La scelta finale del pose estimator per video esterni deve dipendere
dalla disponibilità di coordinate compatibili, dalla qualità osservata e
dal costo computazionale. Non è necessario addestrare un pose estimator
nell'MVP.

------------------------------------------------------------------------

## 12. Valutazione sperimentale

La valutazione deve misurare sia la capacità di ordinare le esecuzioni
per qualità, sia la capacità di rispettare i vincoli delle triplette.

### 12.1 Metriche principali

**Triplet Accuracy**

Percentuale di triplette per cui:

\[ d(A,P)\<d(A,N) \]

È la metrica direttamente collegata all'obiettivo di metric learning.

**Spearman Rank Correlation ((`\rho`{=tex}))**

Misura la correlazione monotona tra distanza predetta e qualità
annotata. Poiché una distanza minore dovrebbe corrispondere a qualità
maggiore, il segno della correlazione va gestito coerentemente: si può
correlare la distanza con il GOE atteso oppure correlare (-d) con il
GOE.

La correlazione deve essere calcolata su campioni indipendenti e con
annotazioni riferite allo stesso elemento.

### 12.2 Analisi aggiuntive

-   distribuzione delle distanze per fasce di GOE;
-   distanza media intra-gruppo e inter-gruppo;
-   andamento della loss su training e validation;
-   prestazioni per elemento tecnico;
-   numero di campioni validi/scartati;
-   sensibilità a rumore e perturbazioni;
-   confronto tra feature base e feature estese.

Se il numero di campioni è ridotto, i risultati devono essere
accompagnati dal numero di osservazioni e interpretati come evidenza
preliminare, non come stima robusta della generalizzazione.

### 12.3 Baseline non neurale

Per giustificare l'utilizzo della GNN, è necessario confrontarla con
almeno una baseline geometrica.

Baseline consigliate:

1.  distanza euclidea tra coordinate canonicalizzate;
2.  distanza tra angoli articolari;
3.  eventualmente cosine similarity tra vettori di feature.

La GNN è giustificata se apprende una metrica più coerente con le
differenze di qualità rispetto a queste misure semplici. Il confronto è
importante perché i nodi sono già semanticamente corrispondenti tra i
grafi.

------------------------------------------------------------------------

## 13. Big Data e gestione computazionale

Il carico principale non è necessariamente il training della GNN, poiché
i grafi con 17 nodi sono piccoli. La parte più costosa può essere il
download, la decodifica e l'estrazione delle pose dai video raw.

La strategia scelta è:

-   privilegiare scheletri già estratti nei dataset accademici;
-   eseguire preprocessing e canonicalizzazione offline;
-   salvare i grafi preprocessati su disco;
-   addestrare la GNN sui grafi serializzati;
-   usare i video raw soltanto per una demo limitata.

Il progetto non prevede l'obbligo di introdurre Spark, Kafka o Hadoop,
poiché nella discussione non sono stati indicati framework specifici dal
corso. La componente Big Data va motivata attraverso la gestione della
pipeline dati, la standardizzazione, il preprocessing e la possibilità
di scalare l'estrazione su grandi raccolte video.

### 13.1 Hardware

Il dispositivo indicato è un Asus Vivobook Pro 15 con processore AMD
Ryzen serie 6000 e GPU NVIDIA, ma il modello esatto della GPU e la VRAM
non sono stati confermati.

Prima di configurare il training:

-   rilevare modello GPU e VRAM;
-   verificare versione CUDA compatibile;
-   installare PyTorch e PyTorch Geometric nelle versioni compatibili;
-   eseguire un test end-to-end su pochi grafi.

Il training deve essere progettato per funzionare anche su CPU o con
batch ridotti. Non si deve basare la pianificazione su una quantità di
VRAM presunta.

------------------------------------------------------------------------

## 14. Struttura software proposta

Una possibile organizzazione del repository:

``` text
figure-skating-aqa/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── graph.yaml
│   └── training.yaml
│
├── data/
│   ├── raw/
│   ├── intermediate/
│   ├── processed/
│   └── splits/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_results_analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── loaders.py
│   │   ├── harmonization.py
│   │   ├── segmentation.py
│   │   └── preprocessing.py
│   │
│   ├── graph/
│   │   ├── topology.py
│   │   ├── features.py
│   │   └── builder.py
│   │
│   ├── training/
│   │   ├── triplet_sampler.py
│   │   ├── augmentation.py
│   │   ├── losses.py
│   │   └── train.py
│   │
│   ├── models/
│   │   ├── encoder.py
│   │   └── siamese_gat.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   ├── inference/
│   │   ├── predict.py
│   │   └── video_pipeline.py
│   │
│   └── visualization/
│       ├── skeleton_plot.py
│       └── attention_plot.py
│
├── checkpoints/
├── reports/
└── tests/
    ├── test_dataset.py
    ├── test_normalization.py
    ├── test_graph.py
    └── test_triplets.py
```

La struttura è una proposta organizzativa. Può essere semplificata se il
tempo disponibile lo richiede, ma è opportuno mantenere separati
caricamento dati, preprocessing, costruzione dei grafi, modello e
valutazione.

------------------------------------------------------------------------

## 15. Dipendenze tecnologiche

Stack previsto:

-   Python;
-   NumPy;
-   SciPy, se necessario per operazioni geometriche;
-   PyTorch;
-   PyTorch Geometric;
-   scikit-learn per utility e metriche;
-   Matplotlib per visualizzazioni;
-   OpenCV per eventuale lettura video;
-   MediaPipe soltanto per la demo, se compatibile.

Le versioni devono essere fissate in `requirements.txt` dopo aver
verificato la compatibilità con Python, GPU e CUDA disponibili.

------------------------------------------------------------------------

## 16. Test e criteri di accettazione

### 16.1 Test della pipeline dati

-   Un file sorgente valido viene caricato senza perdita di metadati.
-   Il numero di joint è verificato e coerente con lo standard atteso.
-   Le coordinate hanno shape e dimensionalità documentate.
-   GOE e segmento temporale sono associati al campione corretto.
-   I campioni invalidi vengono identificati e contabilizzati.

### 16.2 Test della canonicalizzazione

-   Il root-centering porta la root all'origine entro una tolleranza
    numerica.
-   La lunghezza di riferimento del busto diventa unitaria, salvo casi
    invalidi.
-   Lo yaw alignment porta il vettore delle anche all'orientamento
    canonico previsto.
-   La funzione gestisce valori mancanti e denominatori degeneri.
-   La trasformazione non modifica l'ordine o l'identità dei joint.

### 16.3 Test del grafo

-   Ogni grafo contiene 17 nodi.
-   Tutti i grafi usano la stessa topologia.
-   `edge_index` rispetta gli indici dei nodi.
-   `edge_attr` ha dimensione costante e valori finiti.
-   Il grafo è compatibile con il batching PyTorch Geometric.

### 16.4 Test del training

-   I tre rami usano gli stessi parametri.
-   Le triplette rispettano la classe dell'elemento.
-   Positive e Negative rispettano la regola di qualità configurata.
-   La loss è finita e differenziabile.
-   Il training è riproducibile a parità di seed e configurazione.
-   Training e validation sono separati per esibizione/atleta secondo lo
    split definito.

### 16.5 Criteri di successo dell'MVP

L'MVP è considerato implementato quando:

1.  è disponibile un dataset preprocessato e tracciabile;
2.  almeno un elemento tecnico dispone di campioni sufficienti per
    training e valutazione preliminare;
3.  la pipeline produce grafi canonici a 17 nodi;
4.  la Siamese GATv2 completa il training senza errori;
5.  la Triplet Accuracy è superiore al comportamento casuale sul test
    set;
6.  la distanza mostra una relazione interpretabile con il GOE, se le
    annotazioni sono compatibili;
7.  è disponibile un confronto con una baseline geometrica;
8.  viene prodotta almeno una visualizzazione dei pesi di attenzione;
9.  sono documentati limiti, split, iperparametri e risultati.

Non viene fissata una soglia numerica arbitraria di Spearman o Triplet
Accuracy prima di aver analizzato la numerosità e la distribuzione dei
dati. I risultati devono essere riportati integralmente, anche se
negativi.

------------------------------------------------------------------------

## 17. Roadmap operativa di 14 giorni

  -----------------------------------------------------------------------
  Giorni                  Attività                Deliverable
  ----------------------- ----------------------- -----------------------
  1--3                    Download FineFS;        Report esplorativo e
                          analisi `.npz`;         decisione sul dataset
                          verifica GOE, segmenti, 
                          joint e coordinate;     
                          valutazione MMFS        

  4--6                    Selezione elementi;     Dataset PyG
                          split per               preprocessato e testato
                          atleta/esibizione;      
                          canonicalizzazione;     
                          costruzione e           
                          serializzazione grafi;  
                          sampler e augmentation  

  7--10                   Implementazione encoder Checkpoint e report di
                          GATv2; Siamese          training
                          condivisa; Triplet      
                          Loss; training e prime  
                          metriche                

  11--14                  Valutazione test;       MVP, risultati e
                          baseline geometrica;    documentazione
                          attenzione; demo        
                          limitata; report e      
                          presentazione           
  -----------------------------------------------------------------------

### Priorità in caso di ritardi

1.  Completare FineFS e validare il formato dei dati.
2.  Usare un solo elemento tecnico ben rappresentato.
3.  Implementare canonicalizzazione e grafo.
4.  Addestrare la Siamese GATv2 con sampler semplice.
5.  Calcolare le metriche e confrontare una baseline.
6.  Aggiungere MMFS solo se l'integrazione è rapida e verificata.
7.  Aggiungere demo MediaPipe soltanto dopo aver verificato la
    compatibilità delle coordinate.
8.  Completare la visualizzazione dei pesi di attenzione.

La demo non deve compromettere la validazione del modello.

------------------------------------------------------------------------

## 18. Rischi e decisioni ancora aperte

  -----------------------------------------------------------------------
  Rischio / punto aperto  Impatto                 Azione
  ----------------------- ----------------------- -----------------------
  Coordinate FineFS non   Alto                    Ispezionare file e
  comprese o non                                  documentazione prima
  realmente 3D nel                                del preprocessing
  formato atteso                                  

  GOE non associato al    Alto                    Verificare la
  singolo elemento                                granularità delle
                                                  annotazioni; non usarlo
                                                  come label locale se
                                                  non lo è

  Incompatibilità         Alto                    Confrontare schema,
  FineFS-MMFS                                     coordinate, joint e
                                                  annotazioni; escludere
                                                  MMFS se non
                                                  armonizzabile

  Pochi campioni per      Alto                    Ridurre il numero di
  elemento                                        elementi e applicare
                                                  augmentation
                                                  controllata

  Squilibrio tra fasce    Medio/alto              Analizzare
  GOE                                             distribuzioni e
                                                  progettare sampler
                                                  bilanciato

  Overfitting             Alto                    Encoder compatto, split
                                                  per atleta, dropout,
                                                  early stopping

  Differenze di           Alto                    Non effettuare
  coordinate                                      inferenza diretta senza
  HRNet/MediaPipe                                 test di domain
                                                  alignment

  Ambiguità               Medio                   Verificare convenzione
  destra/sinistra                                 dei joint e gestire
                                                  esplicitamente
                                                  eventuali mirror

  Invarianza incompleta   Medio/alto              Documentare limiti e
  al punto di vista                               valutare su dati con
                                                  viste disponibili

  Interpretazione         Medio                   Presentare i pesi come
  impropria                                       indicatori del modello,
  dell'attenzione                                 non come spiegazioni
                                                  causali

  Conversione             Medio                   Usare indice di
  distanza-punteggio non                          similarità e calibrare
  calibrata                                       solo sulla validation
  -----------------------------------------------------------------------

### Decisioni da chiudere prima del training

-   Elementi tecnici definitivi e numerosità minima accettabile.
-   Convenzione esatta dei 17 joint e lista degli archi.
-   Nomi e shape effettivi degli array FineFS.
-   Sistema di riferimento e unità delle coordinate 3D.
-   Granularità e scala dei punteggi GOE.
-   Strategia di aggregazione temporale.
-   Regola per la selezione dei riferimenti.
-   Margine fisso o adattivo della Triplet Loss.
-   Strategia di calibrazione dell'indice di similarità.
-   Compatibilità effettiva della demo MediaPipe.

------------------------------------------------------------------------

## 19. Riferimenti

Riferimenti indicati nella revisione progettuale:

-   FineFS Dataset: https://github.com/yanliji/FineFS-dataset
-   MMFS: https://github.com/dingyn-Reno/MMFS
-   PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
-   GATv2 paper: *How Attentive are Graph Attention Networks?* --- Brody
    et al., 2021.
-   FineGym, citato nella revisione come alternativa per ginnastica:
    https://sdolivia.github.io/FineGym/
-   MMAction2 skeleton data documentation:
    https://github.com/open-mmlab/mmaction2

I riferimenti FineGym e MMAction2 sono mantenuti come
alternative/documentazione metodologica e non fanno parte del dataset
primario scelto per l'MVP.

------------------------------------------------------------------------

## 20. Sintesi dell'architettura finale

``` text
FineFS (.npz)                  MMFS (opzionale)
     |                              |
     +---------- verifica ----------+
                    |
             Data harmonization
                    |
        Selezione elemento + GOE
                    |
          Split per atleta/evento
                    |
          Selezione finestra frame
                    |
        Aggregazione temporale
                    |
       Canonicalizzazione 3D
   (root-centering, scale, yaw)
                    |
          Grafo canonico 17 nodi
          /                   \
   Node features           Edge features
          \                   /
              Siamese GATv2
                    |
             Graph embedding
                    |
          Triplet Margin Loss
          (margine configurabile)
                    |
          Embedding dell'atleta
                    |
       Distanza dal riferimento
                    |
        Indice di similarità
                    |
       Metriche + explainability
```

**Principio guida:** la GNN deve apprendere una metrica di qualità a
partire da grafi coerenti e annotazioni verificabili. La qualità dei
dati, la corretta separazione degli split e la validità della
canonicalizzazione sono prerequisiti del modello, non dettagli
secondari.
