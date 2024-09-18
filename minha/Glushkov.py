"""
Formato do regex possui os caracteres a,b e c.
Todos os operadores devem estar explícitos (., |, e *)
"""

class GlushkovAutomaton:
    def __init__(self, regex): #construtor da classe n
        self.regex = regex #armazena a expressão regular
        self.positions = {}  # Mapeamento da posição de cada caractere
        self.transitions = {}  # Transições do autômato
        self.initial_states = set()
        self.final_states = set()
        self.current_position = 0  # contador que rastreia a posição atual na expressão regular.
        self.build_automaton()  # método para construir autômato
  
    def build_automaton(self):
        """Constrói o autômato baseado na expressão regular."""
        self.parse_expression(self.regex)
        self.determine_final_states()  # Determina os estados finais
  
    def parse_expression(self, expr: str):
        """Analisa a expressão regular e define as transições e como o autômato deve reagir a cada símbolo."""
        previous_position = None  # Variável para rastrear a posição anterior
        for char in expr:
            if char.isalnum():  # Se é um caractere alfanumérico
                self.add_position(char)
                if previous_position is not None:  # Se há uma posição anterior (para concatenação)
                    self.transitions[previous_position].add(self.current_position)
                previous_position = self.current_position  # Atualiza a posição anterior
            elif char == '|':  # Operador de união
                self.handle_union()
                previous_position = None  # Reseta a concatenação
            elif char == '*':  # Operador de estrela de Kleene
                self.handle_kleene_star()
                previous_position = self.current_position  # Para permitir continuação após a estrela de Kleene
            elif char == '.':  # Operador de concatenação explícito
                self.handle_concatenation()


    def add_position(self, char): # Adiciona uma nova posição ao autômato
        """Adiciona uma posição correspondente a um caractere."""
        self.current_position += 1  # incrementa a posição atual
        self.positions[self.current_position] = char
        self.transitions[self.current_position] = set()
        if self.current_position == 1: # adiciona o primeiro elemento como estado inicial
            self.initial_states.add(self.current_position)

    def handle_concatenation(self):
        """Trata o operador de concatenação (.)."""
        if self.current_position > 1:
            last_position = self.current_position - 1  # A transição vai do último estado para o estado atual
            self.transitions[last_position].add(self.current_position)

    def handle_union(self):
        """Trata o operador de união (|)."""
        if len(self.positions) > 1:
            self.initial_states.update(range(1, self.current_position + 1))

    def handle_kleene_star(self):
        """Trata o operador de estrela de Kleene (*)."""
        last = self.current_position
        self.transitions[last].add(last)

    def determine_final_states(self):
        """Determina os estados finais, levando em conta a estrela de Kleene, mas corrige o problema de estados intermediários incorretos."""
        kleene_positions = set()
        self.final_states.clear()  # Certifique-se de limpar qualquer estado final incorreto

        # Primeiro, identificamos os estados que têm uma transição de Kleene (auto-transição)
        for state, targets in self.transitions.items():
            if state in targets:  # Se houver uma transição para si mesmo, temos uma estrela de Kleene
                kleene_positions.add(state)

        # Agora determinamos o último estado real (com um caractere alfanumérico) como estado final principal
        self.final_states.add(self.current_position)  # Último estado é o final padrão

        # Adicionamos outros estados finais possíveis (participantes de Kleene)
        for state in kleene_positions:
            self.final_states.add(state)

        for state in self.positions:
            # Se o estado participa de uma Kleene ou não tem transições saindo, é um estado final
            if state in kleene_positions or not self.transitions[state]:
                self.final_states.add(state)

        # Adiciona o estado anterior a uma Kleene como estado final (caso tenha concatenação antes)
        if self.current_position > 1:
            self.final_states.add(self.current_position - 1)

    def display(self):
        """Exibe o autômato gerado."""
        print("Posições:", self.positions)
        print("Transições:")
        for start, targets in self.transitions.items():
            for target in targets:
                print(f"State({start}, {self.positions[start]}) --{self.positions[start]}--> State({target}, {self.positions[target]})")
        print("Estados Iniciais:", self.initial_states)
        print("Estados Finais:", self.final_states)

# Exemplo de uso
regex = "a.b|c*"
automaton = GlushkovAutomaton(regex)
automaton.display()