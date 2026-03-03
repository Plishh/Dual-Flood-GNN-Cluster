class NodeEdgeConv(MessagePassing):
    """
    This is the core message-passing layer from your implementation.
    It learns to create and propagate messages to update both node and edge features.
    """
    def __init__(self, node_in, node_out, edge_in, edge_out, mlp_layers=2, activation='prelu', device='cpu'):
        super().__init__(aggr='sum')
        
        # MLP for creating messages
        msg_input_size = (node_in * 2 + edge_in)
        self.msg_mlp = make_mlp(input_size=msg_input_size, output_size=edge_out,
                                hidden_size=msg_input_size * 2, num_layers=mlp_layers,
                                activation=activation, device=device)
        
        # MLP for updating edge features after message creation
        edge_update_input_size = (edge_in + edge_out)
        self.edge_update_mlp = make_mlp(input_size=edge_update_input_size, output_size=edge_out,
                                        hidden_size=edge_update_input_size * 2, num_layers=mlp_layers,
                                        activation=activation, device=device)

        # MLP for updating node features after aggregation
        node_update_input_size = (node_in + edge_out)
        self.node_update_mlp = make_mlp(input_size=node_update_input_size, output_size=node_out,
                                        hidden_size=node_update_input_size * 2, num_layers=mlp_layers,
                                        activation=activation, device=device)

    def forward(self, x, edge_index, edge_attr):
        # The custom propagate call will handle message, aggregate, and update steps
        x_out, edge_attr_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return x_out, edge_attr_out

    def propagate(self, edge_index, **kwargs):
        # This custom propagate method is needed because we update both nodes and edges
        x = kwargs['x']
        edge_attr = kwargs['edge_attr']

        # collect all arguments for message passing
        coll_dict = self._collect(self._user_args, edge_index, 'auto', kwargs)

        msg_kwargs = self.inspector.collect_param_data('message', coll_dict)
        messages = self.message(**msg_kwargs)

        aggr_kwargs = self.inspector.collect_param_data('aggregate', coll_dict)
        aggregated_messages = self.aggregate(messages, **aggr_kwargs)

        update_kwargs = self.inspector.collect_param_data('update', coll_dict)
        # Pass 'x' explicitly to the update method
        x_updated = self.update(aggregated_messages, x=x, **update_kwargs)

        # Pass 'edge_attr' explicitly to the edge_update method
        edge_attr_updated = self.edge_update(messages, edge_attr=edge_attr)

        return x_updated, edge_attr_updated

    def message(self, x_i, x_j, edge_attr):
        # Message is a function of source node, target node, and edge features
        return self.msg_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr, x):
        # Node update is a function of its old features and aggregated messages
        return self.node_update_mlp(torch.cat([x, aggr], dim=-1))

    def edge_update(self, msg, edge_attr):
        # Edge update is a function of its old features and the new message
        return self.edge_update_mlp(torch.cat([edge_attr, msg], dim=-1))


class ClusterNodeEdgeGNN(BaseModel):
    """
    Main model class, adapted from your NodeEdgeGNN implementation.
    This structure is suitable for Cluster-GCN training.
    Includes Encoder/Decoder logic from your original file.
    """
    def __init__(self,
                 hidden_features: int = None,
                 num_layers: int = 1,
                 activation: str = 'prelu',
                 residual: bool = True,
                 mlp_layers: int = 2,

                 # Encoder Decoder Parameters
                 encoder_layers: int = 0,
                 encoder_activation: str = None,
                 decoder_layers: int = 0,
                 decoder_activation: str = None,

                 **base_model_kwargs):
        super().__init__(**base_model_kwargs)
        self.with_encoder = encoder_layers > 0
        self.with_decoder = decoder_layers > 0
        self.with_residual = residual


        # Encoder
        encoder_decoder_hidden = hidden_features*2 if hidden_features else self.input_node_features * 2
        if self.with_encoder:
            self.node_encoder = make_mlp(input_size=self.input_node_features, output_size=hidden_features,
                                                hidden_size=encoder_decoder_hidden, num_layers=encoder_layers,
                                            activation=encoder_activation, device=self.device)
            self.edge_encoder = make_mlp(input_size=self.input_edge_features, output_size=hidden_features,
                                                hidden_size=hidden_features, num_layers=encoder_layers, # Edge encoder hidden might differ
                                            activation=encoder_activation, device=self.device)

        # Determine input/output sizes for the GNN core based on encoder/decoder presence
        gnn_input_node_size = hidden_features if self.with_encoder else self.input_node_features
        gnn_output_node_size = hidden_features if self.with_decoder else self.output_node_features
        gnn_input_edge_size = hidden_features if self.with_encoder else self.input_edge_features
        gnn_output_edge_size = hidden_features if self.with_decoder else self.output_edge_features

        self.convs = self._make_gnn(input_node_size=gnn_input_node_size, output_node_size=gnn_output_node_size,
                                    input_edge_size=gnn_input_edge_size, output_edge_size=gnn_output_edge_size,
                                    num_layers=num_layers, mlp_layers=mlp_layers, activation=activation, device=self.device,
                                    hidden_features=hidden_features) # Pass hidden_features for intermediate layers

        # Decoder
        if self.with_decoder:
            self.node_decoder = make_mlp(input_size=hidden_features, output_size=self.output_node_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers,
                                        activation=decoder_activation, bias=False, device=self.device)
            self.edge_decoder = make_mlp(input_size=hidden_features, output_size=self.output_edge_features,
                                        hidden_size=encoder_decoder_hidden, num_layers=decoder_layers, # Edge decoder hidden might differ
                                        activation=decoder_activation, bias=False, device=self.device)

        # Residual connection components
        if self.with_residual:
            # Identity might need adjustment if dimensions change, handled in forward
            self.residual_node = nn.Identity()
            self.residual_edge = nn.Identity()
            self.res_activation = get_activation_func(activation, device=self.device)

    def _make_gnn(self, input_node_size: int, output_node_size: int, input_edge_size: int, output_edge_size: int,
                  hidden_features: int, num_layers: int, mlp_layers: int, activation: str, device: str):

        layers = []
        if num_layers == 1:
            layers.append((
                NodeEdgeConv(input_node_size, output_node_size, input_edge_size, output_edge_size,
                             num_layers=mlp_layers, activation=activation, device=device),
                'x, edge_index, edge_attr -> x, edge_attr'
            ))
        else:
            # First layer: input -> hidden
            layers.append((
                NodeEdgeConv(input_node_size, hidden_features, input_edge_size, hidden_features,
                             num_layers=mlp_layers, activation=activation, device=device),
                'x, edge_index, edge_attr -> x, edge_attr'
            ))
            # Middle layers: hidden -> hidden
            for _ in range(num_layers - 2):
                layers.append((
                    NodeEdgeConv(hidden_features, hidden_features, hidden_features, hidden_features,
                                 num_layers=mlp_layers, activation=activation, device=device),
                    'x, edge_index, edge_attr -> x, edge_attr'
                ))
            # Last layer: hidden -> output (if no decoder) or hidden -> hidden (if decoder)
            final_node_out = output_node_size if not self.with_decoder else hidden_features
            final_edge_out = output_edge_size if not self.with_decoder else hidden_features
            layers.append((
                NodeEdgeConv(hidden_features, final_node_out, hidden_features, final_edge_out,
                             num_layers=mlp_layers, activation=activation, device=device),
                'x, edge_index, edge_attr -> x, edge_attr'
            ))

        return PygSequential('x, edge_index, edge_attr', layers)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # Store initial input for residual connection
        x0, edge_attr0 = x.clone(), edge_attr.clone()

        # 1. Encode features if encoder exists
        if self.with_encoder:
            x = self.node_encoder(x)
            edge_attr = self.edge_encoder(edge_attr)
        
        # Store pre-GNN features for potential residual connection
        x_pre_gnn, edge_attr_pre_gnn = x.clone(), edge_attr.clone()

        # 2. Pass through GNN layers
        x, edge_attr = self.convs(x, edge_index, edge_attr)

        # 3. Decode features if decoder exists
        if self.with_decoder:
            x = self.node_decoder(x)
            edge_attr = self.edge_decoder(edge_attr)

        # 4. Apply residual connection if enabled
        if self.with_residual:
            # Option 1: Add original input (if dimensions match output)
            # This is common if encoder/decoder don't change feature dim drastically
            # or if output_node_features == input_node_features
            if x0.shape == x.shape:
                 x = x + self.residual_node(x0)
            # Option 2: Add pre-GNN state (if dimensions match output)
            # Useful if encoder changes dim but decoder brings it back, or no encoder/decoder
            elif x_pre_gnn.shape == x.shape:
                 x = x + self.residual_node(x_pre_gnn)
            # Apply activation after adding residual
            x = self.res_activation(x)

            # Similar logic for edge attributes
            if edge_attr0.shape == edge_attr.shape:
                 edge_attr = edge_attr + self.residual_edge(edge_attr0)
            elif edge_attr_pre_gnn.shape == edge_attr.shape:
                 edge_attr = edge_attr + self.residual_edge(edge_attr_pre_gnn)
            edge_attr = self.res_activation(edge_attr)


        return x, edge_attr