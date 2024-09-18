from enum import Enum

class GrammarType(Enum):
    UNRESTRICTED = 0
    CONTEXT_SENSITIVE = 1
    CONTEXT_FREE = 2
    REGULAR = 3

def sinalpha(s: str, alpha: set) -> bool:
    """
    Testa se todas as letras de s pertencem
    ao alfabeto alpha.
    """
    return all(a in alpha for a in s)

class gg:
    """
    Generic grammar.
    """

    def __init__(self) -> None:
        self.terms = set()
        self.nonterms = set()
        self.init = None
        self.prod = set()
        self.type = GrammarType.UNRESTRICTED

    @classmethod
    def fromfile(cls, myg: str):
        g = cls()
        with open(myg) as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith("#")]

            # Alfabetos
            g.terms = set(lines.pop(0).strip())
            g.nonterms = set(lines.pop(0).strip())

            # Símbolo inicial e produções
            g.init = lines.pop(0).strip()
            for line in lines:
                left, right = line.strip().split(",")
                g.prod.add((left, right))
                if not sinalpha(left, g.terms | g.nonterms):
                    raise Exception("Lado esquerdo de regra com letras não declaradas.")
                if not sinalpha(right, g.terms | g.nonterms):
                    raise Exception("Lado direito de regra com letras não declaradas.")

        if g.iscs():
            g.type = GrammarType.CONTEXT_SENSITIVE

        return g

    def iscs(self) -> bool:
        """
        Testa se a gramática é sensível ao contexto.
        Uma gramática é sensível ao contexto se, para cada produção A -> B,
        temos que o tamanho de B é maior ou igual ao tamanho de A.
        """
        for left, right in self.prod:
            if len(left) > len(right):
                return False
        return True

    def iscfg(self) -> bool:
        """
        Testa se a gramática é livre de contexto.
        Uma gramática é livre de contexto se, para cada produção,
        o lado esquerdo é um único símbolo não terminal.
        """
        for left, right in self.prod:
            if len(left) != 1 or left not in self.nonterms:
                return False
        return True

    def isreg(self) -> bool:
        """
        Testa se a gramática é regular.
        Uma gramática é regular se todas as produções têm a forma
        A -> aB ou A -> a, onde A e B são não terminais e a é um terminal.
        """
        for left, right in self.prod:
            if len(left) != 1 or left not in self.nonterms:
                return False
            if len(right) > 2 or (len(right) == 2 and right[1] not in self.nonterms):
                return False
        return True

    def lang(self, s: str) -> bool:
        """
    Testa se a cadeia 's' pertence à linguagem gerada pela gramática.
    max_steps: Limita o número de passos para evitar loops infinitos.
    """
    tape = [self.init]
    steps = 0

    while tape and steps < max_steps:
        u = tape.pop(0)
        steps += 1
        sprods = []
        
        # Verificar se a cadeia gerada é muito maior que a cadeia de entrada
        if len(u) > len(s):
            continue  # Descartar esta derivação, já que ela não pode ser s
        
        # Tentar substituir todas as partes da cadeia
        for p in range(len(u)):
            for left, right in self.prod:
                if u[p:].startswith(left):  # Se a subcadeia a partir de p inicia com 'left'
                    # Substituir 'left' por 'right' na cadeia
                    v = u[:p] + right + u[p + len(left):]
                    if v == s:  # Se a cadeia é igual à desejada    
                        return True
                    sprods.append(v)  # Adicionar nova derivação à lista

        # Adicionar novas derivações à fita se estiverem progredindo
        tape.extend(sprods)
        return False  # Se o limite de passos foi atingido ou nenhuma derivação foi aceita

if __name__ == "__main__":
    myg = r"anbncn.grm"

    g = gg.fromfile(myg)
    print(g.prod)
    if g.lang("aaabbbccc"):    
        print("Na linguagem")
    else:
        print("Fora da linguagem")
