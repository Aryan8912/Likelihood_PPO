import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Dummy data and parameters for demonstration
device = "cuda" if torch.cuda.is_available() else "cpu"
num_prompts = 100
seq_len = 20
vocab_size = 1000
embedding_dim = 64
hidden_dim = 128
num_layers = 2
batch_size = 16
num_epochs = 5
learning_rate = 1e-3
num_samples_n = 4  # N in the algorithm
verifier_lambda_n = 0.1  # Lambda_N (weight of the negative sampling term)

# Dummy expert dataset (prompts and solutions as sequences of indices)
expert_prompts = torch.randint(0, vocab_size, (num_prompts, seq_len)).to(device)
expert_solutions = torch.randint(0, vocab_size, (num_prompts, seq_len)).to(device)
expert_dataset = TensorDataset(expert_prompts, expert_solutions)
expert_dataloader = DataLoader(expert_dataset, batch_size=batch_size, shuffle=True)

# Dummy verifier score function (replace with your actual verifier)
def dummy_verifier_score(prompt, response):
    # A simple score based on the number of matching tokens (for demonstration)
    match = (response == prompt).sum(dim=1).float() / prompt.size(1)
    return match.unsqueeze(1)

# Simple Language Model (replace with your actual LLM)
class SimpleLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        logits = self.fc(output)
        return logits

model = SimpleLM(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def loss_fn(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

def calculate_f_theta_gradient(model, prompt, solution, generated_responses, verifier, lambda_n):
    """Approximation of the gradient of f_theta."""
    model.train()
    optimizer.zero_grad()

    # Gradient of log likelihood of the expert solution
    expert_logits = model(prompt)
    expert_log_prob = -loss_fn(expert_logits[:, :-1, :], solution[:, 1:])
    expert_log_prob.backward(retain_graph=True)
    grad_expert_log_prob = [param.grad.clone() for param in model.parameters()]
    model.zero_grad()

    # Expectation over generated responses (approximation with N samples)
    expected_grad = [torch.zeros_like(param).to(device) for param in model.parameters()]
    for gen_response in generated_responses:
        gen_logits = model(prompt)
        gen_log_prob = -loss_fn(gen_logits[:, :-1, :], gen_response[:, 1:])
        reward_expert = verifier(prompt, solution).mean()
        reward_gen = verifier(prompt, gen_response).mean()
        weight = torch.sigmoid(reward_expert - reward_gen).item()  # Use .item() to get a scalar

        gen_log_prob.backward(retain_graph=True)
        for i, param in enumerate(model.parameters()):
            expected_grad[i] += weight * param.grad.clone()
        model.zero_grad()

    for i in range(len(expected_grad)):
        expected_grad[i] /= len(generated_responses)

    # Combine the gradients
    final_grad = [expert_grad - lambda_n * expected_grad_item for expert_grad, expected_grad_item in zip(grad_expert_log_prob, expected_grad)]

    return final_grad

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(expert_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (prompts, solutions) in enumerate(progress_bar):
        optimizer.zero_grad()
        batch_size_current = prompts.size(0)
        total_batch_loss = 0

        for i in range(batch_size_current):
            prompt = prompts[i].unsqueeze(0)
            solution = solutions[i].unsqueeze(0)

            # Step 5: Sample N responses
            generated_responses = [model.generate(prompt, max_length=seq_len) for _ in range(num_samples_n)]

            # Step 6: Select the BoN response
            with torch.no_grad():
                scores = torch.cat([dummy_verifier_score(prompt, res) for res in generated_responses], dim=0)
                best_response_index = torch.argmax(scores)
                bon_response = generated_responses[best_response_index]

            # Step 7: Compute the gradient (approximated using the loss function)
            expert_logits = model(prompt)
            expert_loss = loss_fn(expert_logits[:, :-1, :], solution[:, 1:])

            bon_logits = model(prompt)
            bon_loss = loss_fn(bon_logits[:, :-1, :], bon_response[:, 1:])

            # Step 9: Update parameters (simplified gradient update)
            loss = expert_loss - bon_loss # Simplified version of the gradient update
            loss.backward()
            total_batch_loss += loss.item()

        optimizer.step()
        total_loss += total_batch_loss / batch_size_current
        progress_bar.set_postfix({"loss": total_loss / (batch_idx + 1)})

    print(f"Epoch {epoch+1} Loss: {total_loss / len(expert_dataloader)}")

# Evaluation (simple generation for demonstration)
model.eval()
with torch.no_grad():
    test_prompt = expert_prompts[0].unsqueeze(0)
    generated_output = model.generate(test_prompt, max_length=seq_len)
    print("Test Prompt:", test_prompt)
    print("Generated Output:", generated_output)