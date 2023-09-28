import torch
import torch.nn as nn

class PosEnc(nn.Module):
    def __init__(self, d_input, n_freqs, log_space=False) -> None:
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = self.d_input * (1 + 2 * n_freqs)
        self.embed_fns = [lambda x: x]

        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x):
        out = []
        for embed_fn in self.embed_fns:
            out.append(embed_fn(x))

        return torch.concat(out, dim=-1)


class NeRF(nn.Module):
    def __init__(self, d_input=3, n_layers=8, d_filter=256, skip=[4], n_freqs=10, log_space=True, n_freqs_views = 0):
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        
        self.posenc = PosEnc(d_input, n_freqs=n_freqs, log_space=log_space)
        self.encode_view = False
        if n_freqs_views > 0:
            self.view_posenc = PosEnc(d_input, n_freqs=n_freqs_views, log_space=log_space)
            self.encode_view = True

        # Create model layers
        self.layers = nn.ModuleList()
        # the first layer input dimension should be after positional encoding
        self.layers.append(nn.Linear(self.posenc.d_output, d_filter))
        for i in range(1, n_layers - 1):
            if i in self.skip:
                # adding a skip connect in the middle
                self.layers.append(nn.Linear(d_filter + self.posenc.d_output, d_filter))
            else:
                self.layers.append(nn.Linear(d_filter, d_filter))                

        # Bottleneck layers
        if self.encode_view:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.view_posenc.d_output, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, 4)

    def forward(self, x):       
        x_input = x
        x = self.posenc(x)
        x_pos = x
        for i, layer in enumerate(self.layers):
            if i in self.skip:
                x = torch.cat([x, x_pos], dim=-1)            
            x = layer(x)
            x = self.act(x)

        # Apply bottleneck
        if self.encode_view:
            # Split alpha from network output
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            if self.encode_view:
                x = torch.concat([x, self.view_posenc(x_input)], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x