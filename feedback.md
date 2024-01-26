# Feedback

Hi Giskard 🐢,

As requested in your technical test, here are my feedback about my experience with Giskard, some ideas for future improvement, and considerations about how it may or may not work in my usual workflow.

### Generalities

I really like the fact that the Model class wrapper is actually model-agnostic, and that I need to make no further assumptions than "It takes a `pd.DataFrame` and returns a 1-D array or pd.Series". It is really easy to use, the interfaces and APIs are easy to understand and nicely arranged, and work as I would imagine they would. Giskard brings a lot of value after a few minutes of usage, thanks to `giskard.scan()`.

This is of key importance to me, as it means that I won't have a severe adherence to Giskard as a dependency. We can simply add a new step at the end of an existing training pipeline codebase to perform the tests on a newly trained model.

### Deploying the Giskard Hub - A word on Dev/MLOps

As per my understanding, `giskard hub start` runs a Docker container via the `docker` Python API.

However, this makes a very strong assumption about the way to deploy the hub: the environment one wants to deploy Giskard onto is not already containerized (i.e., VM or bare metal server).

In my restricted view of the current ML stack, the best practice is to containerize everything, and everything is already a container. In Vertex AI Notebooks, the Jupyter kernels are actually containers. As a consequence, I could not even run the hub within my environment (or I could, but I'd have to handle Docker inside Docker, which is not in my skill set).

As a Data Scientist, this would be my workflow when using Giskard:

- Either I want to experiment with Giskard, discover its value, while having no impact on my production environment. In this case, in my notebooks, I would like to instantiate a hub by myself, in an mlflow standalone fashion. As such, I expect the `giskard hub start` command to simply start a new process with no Docker involved (again, because I'm already in a container).

- Or, I am in a more advanced, industrialized setup, and I want to deploy Giskard my own way. I want to deploy the hub in my own fashion (in a serverless setup like Cloud Run, or in a Kubernetes cluster). In this scenario, I need to have direct access to the Giskard container (hub + ML workers), with documentation for the arguments and options.

In addition, Giskard could propose template Kubernetes manifests, a template Kubeflow/VertexAI pipeline using Giskard, or even architectural patterns to implement.

Finally, the Giskard version provided in the Hugging Face template looks a bit outdated. I was a bit surprised that I had to downgrade my local dependency from `2.5.3` to `2.1.2`.

### Feedback on Giskard

I am convinced that, whether there is a hype around the whole LLM trend or not, a rough 90% of the machine learning use case in the industry is the usual, "old-fashioned" tabular dataset use case. I like the fact that you did not overlook this part.

I was lacking the ability to also include my data preprocessing steps, i.e., `featurizer.run()` in the Giskard `Model()`, because it expects the same column names (or a subset) than the `Dataset()`, which I renamed. This could be an important feature because, from experience, data leakage usually appears in such processing. See `featurizer.get_previous_arrival_delay()`, where I add the delay of the previous flight. What if I did it the other way around and added the delay of the next flight?

Now that the Giskard library is quite stable, and most of its core markets are addressed, here are some more features I think are currently missing:

- A SparkML and spark dataframes integration in Giskard. There may definitely be a hacky way on the user side to achieve this with `spark_df.toPandas()`, but still, a correct integration would be awesome for SparkML users.
- "Non-standard" ML models such as time-series (especially with exogenous data), recommender systems, or unsupervised models.
- The categorical selection drop-down menu in the Giskard hub debugger should be sorted alphabetically, and a filter/search box would be a great addition.

I was also a bit disappointed by the documentation regarding integration with public cloud providers. I already talked about the integration with the hub earlier. Additionally, Giskard should definitely propose a way to integrate cloud providers' ML services and/or data sources (e.g., integration with Vertex Datasets or Model).

### Documentation

I think the quickstart part should include more "concept-oriented" pages that explain the several concepts behind Giskard (e.g., `giskard.scan()`, `giskard.testing`, `@transformation_function`, or `@slicing_function`), rather than a use-case oriented (LLM, tabular, NLP) tour.

Additionally, the [Dataset documentation](https://docs.giskard.ai/en/latest/reference/datasets/index.html#giskard.Dataset.data_processor) refers to a `DataProcessor` class that I could not find in the documentation. I was a bit confused until I found this was an internal class used for slices and transformations. So either make it public and document it, or make it more trivial what its usage is.