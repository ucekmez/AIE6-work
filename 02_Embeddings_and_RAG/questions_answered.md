#### ❓Question #1:

The default embedding dimension of `text-embedding-3-small` is 1536, as noted above. 

1. Is there any way to modify this dimension?
2. What technique does OpenAI use to achieve this?

#### Answer #1:

1. Yes, the api has `dimensions` input to update model dimension
2. It's called `Matryoshka`


#### ❓Question #2:

What are the benefits of using an `async` approach to collecting our embeddings?

#### Answer #2:

It allows us to use internal threads when we don't want to wait for the actual result of a given function call. Since we calculate a lot of embeddings from a given doc, we can utilize `async` to fire multiple embedding calls via async methods and then continue the program lines after all embeddings are completed. This decreases the overhead of syncronous approach.


#### ❓ Question #3:

When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?

#### Answer #3:

We can tweak the responses via temperature parameter to increase or decrease the randomness/creativity while generating a response.

#### ❓ Question #4:

What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?

What is that strategy called?

#### Answer #4:

There are a few days to let agent be more thoughtful. One way is a strategy is called `chain of thought` prompting. It's to give LLM some instructions in the system prompt by simply saying `think step-by-step before responding, work through this problem one step at a time` or similar. Or we can just let it return the response by including the reasoning process before giving the final answer in the response. 