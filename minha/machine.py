from __future__ import annotations

import copy
import enum
import random
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from generative import simpleRE


@dataclass
class runresult:
    accept: bool
    steps: int
    computation: List[Tuple[str, int, str]]
    state: str


##############
### States ###
##############


class absstate(ABC):
    """
    A generic state class for automata and other machines.
    """

    def __init__(self, dad, id: str, label: str = None, data=None) -> None:
        self.id = id
        self.data = data  # Any data associated with the state.
        self.dad = dad  # Machine to which the state belongs.
        self.label = (
            label if label else id
        )  # Printable name, by default is the state id
        # Dictionaries of transitions, indexed by labels.
        self.incoming = {}  # Incoming transitions
        self.outgoing = {}  # Outgoing transitions

    def __str__(self):
        return self.id

    @abstractmethod
    def adj(self, x: str):
        """
        Returns the ends of outgoing transitions whose input label is x.
        """
        pass

    def incomingset(self) -> set:
        """
        Returns the set of all incoming transitions.
        """
        return set(self.incoming.values())

    def outgoingset(self) -> set:
        """
        Returns the set of all outgoing transitions.
        """
        setsoftransitions = list(self.outgoing.values())
        return set() if not setsoftransitions else set.union(*setsoftransitions)


class nfastate(absstate):
    """
    States for nondeterministic finite automata.
    """

    def __init__(self, dad, id: str, label: str = None, data=None) -> None:
        super().__init__(dad, id, label, data)

    def adj(self, a: str) -> set:
        """
        Returns the set of outgoing transitions labelled by the letter a.
        """
        return set([t.end for t in self.outgoing if t.input == a])

class pdastate(absstate):
    """
    States for nondeterministic pushdown automata.
    """

    def __init__(self, dad, id: str, label: str = None, data=None) -> None:
        super().__init__(dad, id, label, data)

    def adj(self, a: str) -> set:
        """
        Returns the set of outgoing transitions labelled by the letter a.
        """
        return set([t.end for t in self.outgoing if t.input == a])


class dpdastate(absstate):
    """
    States for deterministic pushdown automata.
    """

    def __init__(self, dad, id: str, label: str = None, data=None) -> None:
        super().__init__(dad, id, label, data)

    def adj(self, a: str) -> dfastate:
        """
        Returns the end of the outgoing transition labelled by the letter a.
        None if no transitions is labelled by a.
        """
        if a not in self.outgoing:
            return None
        return self.outgoing[a].end

    def defact(self, a: str, q: dfastate) -> None:
        """
        Defines the outgoing transition labelled by the letter a as self -> q.
        An exception is raised if a is not a letter.
        """
        atransition(self, q, a)

    def act(self, x: str) -> dfastate:
        """
        Returns the state reached by the reading of x from this state.
        """
        if x == "":
            return self
        a = x[0]
        q = self.adj(a)
        if not q:
            return q
        return q.act(x[1:])


class dtmstate(absstate):
    """
    States for deterministic Turing machines.
    """

    pass


###################
### Transitions ###
###################


class abstransition(ABC):
    """
    Common data for all types of transitions.
    """

    def __init__(self, start: absstate, end: absstate, input=None) -> None:
        self.input = input
        if start.dad != end.dad:
            raise Exception("Ends of transitions must belong to the same machine.")
        self.chstart(start)
        self.chend(end)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def chstart(self, start) -> None:
        """
        Changes start state of the transition.
        This should be implemented in every subclass.
        """
        pass

    @abstractmethod
    def chend(self, end) -> None:
        """
        Changes end state of the transition.
        This should be implemented in every subclass.
        """
        pass

    def getstart(self) -> str:
        """
        Returns the string id of the start state.
        """

        return str(self.start)

    def getend(self) -> str:
        """
        Returns the string id of the end state.
        """

        return str(self.end)


class atransition(abstransition):
    """
    Finite automaton transition (labelled by a letter)
    """

    def __init__(self, start: absstate, end: absstate, input: str):
        """
        Builds a new transition labelled by the letter input.
        The transition is created from absstate objects, not string ids.
        If input is not a letter, an exception is thrown.
        """
        if len(input) != 1:
            raise Exception("Label should be a letter.")
        if input not in start.dad.inalpha:
            start.dad.inalpha.add(input)
        super().__init__(start, end, input)

    def __str__(self):
        return f"({self.start}, {self.input}, {self.end})"

    def chstart(self, start) -> None:
        self.start = start
        if isinstance(start, dfastate):
            # If start is a DFA state, then its outgoing dictionary
            # associates labels to transitions.
            start.outgoing[self.input] = self
        else:
            # Otherwise, it associate labels to sets of transitions.
            if self.input not in start.outgoing:
                start.outgoing[self.input] = {
                    self,
                }
            else:
                start.outgoing[self.input].add(self)

    def chend(self, end) -> None:
        self.end = end
        if self.input not in end.incoming:
            end.incoming[self.input] = {
                self,
            }
        else:
            end.incoming[self.input].add(self)


class pdatransition(abstransition):
    def __init__(
        self,
        start: absstate,
        end: absstate,
        input: str = None,
        pop: str = None,
        push: str = None,
    ) -> None:
        super().__init__(start, end, input)
        self.push = push
        self.pop = pop

    def __str__(self):
        return f"({self.start}, {self.input}, {self.pop}, {self.push}, {self.end})"


class tmtransition(abstransition):
    """
    Turing machine transition
    """

    def __init__(
        self, start: absstate, end: absstate, input: str, output: str, direction: str
    ) -> None:
        super().__init__(start, end, input)
        self.output = output
        self.direction = direction
        if direction not in (
            abstractmachine.notations.LEFT,
            abstractmachine.notations.RIGHT,
        ):
            raise Exception(
                "Directions should be "
                + abstractmachine.notations.LEFT
                + " or "
                + abstractmachine.notations.RIGHT
            )


################
### Machines ###
################


class abstractmachine(ABC):
    def __init__(self, inalpha="", outalpha="", auxalpha="", auxdevice=None) -> None:
        self.states = set()
        self.initial = None
        self.accept = None
        self.reject = None
        self.inalpha = set(inalpha)
        self.outalpha = set(outalpha)
        self.auxalpha = set(auxalpha)
        self.auxdevice = auxdevice
        self.idcount = 0
        self.statesdict = {}

    class notations(enum.Enum):
        LEFT = "E"
        RIGHT = "D"
        BLANK = "_"

    @classmethod
    def blank(cls):
        return "_"

    @classmethod
    @abstractmethod
    def fromfile(cls, spec: str) -> abstractmachine:
        """
        Creates a new machine object from a file specification.
        """
        pass

    @abstractmethod
    def printmachine(self) -> None:
        """
        Prints a readable description of the machine.
        """
        pass

    @abstractmethod
    def transitions(self) -> set:
        """
        Set of transitions of the machine.
        """
        pass

    def state(self, id: str) -> absstate:
        """
        Returns the state object from an id, None if there is no such state.
        """

        if id not in self.statesdict:
            return None
        return self.statesdict[id]

    def n(self) -> int:
        """
        Number of states of the machine.
        """
        return len(self.states)

    def m(self) -> int:
        """
        Number of transitions of the machine.
        """
        return len(self.transitions())

    def stateset(self) -> set:
        return set([str(q) for q in self.states])

    @abstractmethod
    def addstate(self, id: str = None) -> str:
        """
        This method is intended to add a new state identified by id.
        If no id is provided, a new one is obtained from a counter.
        Subclasses must create the corresponding state object,
        and update self.states and self.statesdict accordingly.
        The id (a string) is returned.
        """
        if not id:
            id = str(self.idcount)
            self.idcount += 1
        if id in self.statesdict:
            raise Exception("State id clash.")
        return id

    def addstates(self, *ids) -> None:
        """
        This method adds any number of states,
        identified by their ids.
        """
        for id in ids:
            self.addstate(id)

    @abstractmethod
    def addtransition(self, startid: str, endid: str, input, output=None, dir=None):
        pass

    @abstractmethod
    def addbouquet(self, *bouquets) -> None:
        pass

    def draw(self):
        G = nx.MultiDiGraph()

        for q in self.stateset():
            G.add_node(q)

        for e in self.transitions():
            G.add_edge(e.getstart(), e.getend(), label=e.input)

        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_size=700)

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=2000,
            node_color="lightblue",
            arrows=True,
            arrowstyle="->",
            arrowsize=10,
            width=2,
            connectionstyle="arc3,rad=0.3",
        )
        labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.savefig("output.png")


class nfa(abstractmachine):
    """
    Nondeterministic finite automata.
    """

    def __init__(self, inalpha="", outalpha="", auxalpha="") -> None:
        super().__init__(inalpha, outalpha, auxalpha)
        self.initial = set()
        self.accept = set()

    @classmethod
    def fromfile(cls, spec: str) -> nfa:
        A = cls()

        with open(spec) as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith("#")]

            # Alphabet
            A.inalpha = set(lines.pop(0).strip())

            # Initial/Final
            for q in lines.pop(0).strip().split(","):
                id = A.addstate(q)
                A.setinitial(q)
            for q in lines.pop(0).strip().split(","):
                id = q if q in A.initial else A.addstate(q)
                A.setfinal(id)

            # Transitions
            for line in lines:
                p, a, q = line.strip().split(",")
                A.addtransition(p, q, a)

        return A

    def printmachine(self) -> None:
        print("alphabet: ", self.inalpha)
        print("states: ", self.stateset())
        print("initial states: ", self.initials())
        print("final states: ", set([str(q) for q in self.accept]))
        print("transitions: ", set(map(str, self.transitions())))

    def addstate(self, id: str = None, label: str = None, data=None) -> str:
        id = super().addstate(id)
        s = nfastate(self, id, label, data)
        self.states.add(s)
        self.statesdict[id] = s
        return id

    def transitions(self) -> set:
        """
        Set of transitions of the automaton.
        """
        S = [q.outgoingset() for q in self.states]
        return set.union(*S)

    def addtransition(self, startid: str, endid: str, input, output=None, dir=None):
        p = self.state(startid)
        q = self.state(endid)
        if not p:
            p = self.state(self.addstate(startid))
        if not q:
            q = self.state(self.addstate(endid))
        atransition(p, q, input)

    def addbouquet(self, *bouquets) -> None:
        """
        Adds an arbitrary number of bouquets.
        A bouquet is a string of form q0 a1 q1 a2 q2 ... ak qk
        where q0, q1, ..., qk are states and a1, a2, ..., ak
        are letters. The transitions (q0, a1, q1),
        (q0, a2, q2), ..., (q0, ak, qk) are added.
        """

        for s in bouquets:
            parts = s.split()
            p = parts.pop(0)
            while parts:
                a, q = parts.pop(0), parts.pop(0)
                self.addtransition(p, q, a)

    def setfinal(self, id: str) -> None:
        """
        Sets state id as final.
        """
        q = self.state(id)
        if not q:
            raise Exception("State " + id + " does not exist.")
        self.accept.add(q)

    def setinitial(self, id: str) -> None:
        """
        Sets state id as initial.
        """
        q = self.state(id)
        if not q:
            raise Exception("State " + id + " does not exist.")
        self.initial.add(q)

    def finals(self) -> set:
        """
        Returns the set of final states
        """
        return self.accept

    def initials(self) -> set:
        """
        Returns the set of initial states
        """
        return set([str(q) for q in self.initial])


class dfa(abstractmachine):
    """
    Deterministic finite automata.
    """

    def __init__(self, inalpha="", outalpha="", auxalpha="") -> None:
        super().__init__(inalpha, outalpha, auxalpha)
        self.accept = set()

    def printmachine(self) -> None:
        print("alphabet: ", self.inalpha)
        print("states: ", self.stateset())
        print("initial state: ", self.initial)
        print("final states: ", set([str(q) for q in self.accept]))
        print("transitions: ", set(map(str, self.transitions())))

    @classmethod
    def fromfile(cls, spec: str) -> dfa:
        A = cls()

        with open(spec) as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith("#")]

            # Alphabet
            A.inalpha = set(lines.pop(0).strip())

            # Initial/Final
            initial = lines.pop(0).strip()
            A.addstate(initial)
            A.setinitial(initial)
            for q in lines.pop(0).strip().split(","):
                id = initial if q == initial else A.addstate(q)
                A.setfinal(id)

            # Transitions
            for line in lines:
                p, a, q = line.strip().split(",")
                A.addtransition(p, q, a)

        return A

    def transitions(self) -> set:
        """
        Set of transitions of the automaton.
        """
        S = [q.outgoingset() for q in self.states]
        return set.union(*S)

    def addstate(self, id: str = None, label: str = None, data=None) -> str:
        id = super().addstate(id)
        s = dfastate(self, id, label, data)
        self.states.add(s)
        self.statesdict[id] = s
        return id

    def addtransition(self, startid: str, endid: str, input, output=None, dir=None):
        p = self.state(startid)
        q = self.state(endid)
        if not p:
            p = self.state(self.addstate(startid))
        if not q:
            q = self.state(self.addstate(endid))
        p.defact(input, q)

    def addbouquet(self, *bouquets) -> None:
        """
        Adds an arbitrary number of bouquets.
        A bouquet is a string of form q0 a1 q1 a2 q2 ... ak qk
        where q0, q1, ..., qk are states and a1, a2, ..., ak
        are letters. The transitions (q0, a1, q1),
        (q0, a2, q2), ..., (q0, ak, qk) are added.
        """

        for s in bouquets:
            parts = s.split()
            p = parts.pop(0)
            while parts:
                a, q = parts.pop(0), parts.pop(0)
                self.addtransition(p, q, a)

    def setfinal(self, id: str) -> None:
        """
        Sets state id as final.
        """
        q = self.state(id)
        if not q:
            raise Exception("State " + id + " does not exist.")
        self.accept.add(q)

    def setinitial(self, id: str) -> None:
        """
        Sets state id as initial.
        """
        q = self.state(id)
        if not q:
            raise Exception("State " + id + " does not exist.")
        self.initial = q

    def finals(self) -> set:
        """
        Returns the set of final states
        """
        return self.accept

    @classmethod
    def product(cls, A: dfa, B: dfa) -> dfa:
        """
        Builds the Cartesian product of two DFA.
        No final state is set.
        """

        C = cls()

        states = [(p, q) for p in A.states for q in B.states]
        ids = list(map(lambda pair: f"({pair[0].id}, {pair[1].id})", states))

        for k in range(len(states)):
            C.addstate(ids[k], data=states[k])
            p, q = states[k]
            if (A.initial == p) and (B.initial == q):
                C.setinitial(ids[k])

        transitions = [
            (e, f)
            for e in A.transitions()
            for f in B.transitions()
            if e.input == f.input
        ]
        for e, f in transitions:
            C.addtransition(
                f"({e.start.id}, {f.start.id})", f"({e.end.id}, {f.end.id})", e.input
            )

        return C

    def act(self, input: str) -> dfastate:
        """
        End state for the computation reading input.
        """

        return self.initial.act(input)


class pda(abstractmachine):
    def __init__(self, inalpha="", outalpha="", auxalpha="", auxdevice=None) -> None:
        super().__init__(inalpha, outalpha, auxalpha, auxdevice)
        self.accept = set()


class tm(abstractmachine):
    """
    Máquinas de Turing determinísticas, usando a definição do livro de Sipser
    """

    def printmachine(self) -> None:
        print(self.states)
        print(self.initial)
        print(self.accept)
        print(self.reject)
        print(self.inalpha)
        print(self.auxalpha)
        print(self.T)

    @classmethod
    def fromfile(cls, mytm: str):
        t = cls()

        with open(mytm) as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith("#")]

            # Alfabetos
            t.inalpha = set(lines.pop(0).strip())
            t.auxalpha = set(lines.pop(0).strip())
            t.auxalpha = t.inalpha | t.auxalpha
            t.auxalpha.add(cls.notations.BLANK)

            # Estados especiais
            t.initial = lines.pop(0).strip()
            t.accept = lines.pop(0).strip()
            t.reject = lines.pop(0).strip()
            t.states.add(t.initial)
            t.states.add(t.accept)
            t.states.add(t.reject)

            for line in lines:
                p, a, b, dir, q = line.strip().split(",")
                t.T[(p, a)] = (b, dir, q)
                t.states.add(p)
                t.states.add(q)
                if a not in t.auxalpha:
                    raise Exception("Transição com letra não declarada.")

        return t

    def run(self, tape: str, steps: int = -1) -> runresult:
        """
        Executa a máquina com a fita de entrada dada pela string tape.
        O parâmetro opcional steps estipula a quantidade máxima de passos da execução.
        """

        r = runresult(False, 0, [], self.initial)

        tape = deque(tape)
        pos = 0

        r.computation.append((copy.deepcopy(tape), pos, r.state))

        # print(r.state)
        # print("".join(tape))
        # print("^")

        while r.steps != steps:  # Se steps não for definido, aqui sempre será True
            t, dir, q = self.T[(r.state, tape[pos])]
            tape[pos] = t
            r.state = q
            if dir == "D":
                pos += 1
                if pos == len(tape):
                    tape.append("_")
            if dir == "E":
                pos -= 1
                if pos < 0:
                    tape.appendleft("_")
                    pos = 0
            r.steps += 1
            r.computation.append((copy.deepcopy(tape), pos, r.state))
            if r.state == self.accept:
                r.accept = True
                return r
            if r.state == self.reject:
                r.accept = False
                return r

            # print(r.state)
            # print("".join(tape))
            # print(" " * pos + "^")

    def computation(self, tape: str, steps: int = -1) -> None:
        """
        Executa a máquina com a entrada "tape", por no máximo
        "steps" passos (se definido), em seguida imprime a
        sequência de configurações da computação.
        """

        r = self.run(tape, steps)

        for t, pos, state in r:
            print(state)
            print("".join(t))
            print(" " * pos + "^")

    def complexity(
        self, tapes: List[str], f=None, label=None
    ) -> Tuple[List[int], List[int]]:
        """
        Recebe uma lista "tapes" de cadeias, entendidas como entradas para a
        máquina de Turing, executa a máquina em cada cadeia da lista e
        gera um par de listas de inteiros, representando respectivamente
        os comprimentos das cadeias em "tapes" e os números de passos
        da execução da máquina.
        Se f for definido, assume-se que é uma função numérica. Neste caso,
        é produzida uma plotagem dessa função, juntamente das listas de pontos
        geradas. Se f for definido, assume-se que label também é, e que
        representa a legenda da função f na plotagem.
        """

        x = []
        y = []
        for s in tapes:
            x.append(len(s))
            r = self.run(s)
            y.append(r.steps)

        if f:
            plt.plot(x, y, ".", label="TM")
            plt.xlabel("comprimento da cadeia")
            plt.ylabel("passos da MT")
            plt.title("Complexidade de uma TM")
            u = np.linspace(1, x[-1], 200)
            v = f(u)
            plt.plot(u, v, label=label)
            plt.legend()
            plt.grid(True)
            plt.show()

        return (x, y)


class ntm(abstractmachine):
    """
    Máquinas de Turing não-determinísticas, usando a definição do Sipser
    """

    def printmachine(self) -> None:
        print(self.states)
        print(self.initial)
        print(self.accept)
        print(self.reject)
        print(self.inalpha)
        print(self.auxalpha)
        print(self.T)

    @classmethod
    def fromfile(cls, mytm: str):
        t = cls()

        with open(mytm) as f:
            lines = [line.strip() for line in f.readlines() if not line.startswith("#")]

            # Alfabetos
            t.inalpha = set(lines.pop(0).strip())
            t.auxalpha = set(lines.pop(0).strip())
            t.auxalpha = t.inalpha | t.auxalpha
            t.auxalpha.add(cls.blank())

            # Estados especiais
            t.initial = lines.pop(0).strip()
            t.accept = lines.pop(0).strip()
            t.reject = lines.pop(0).strip()
            t.states.add(t.initial)
            if t.accept != "_":
                t.states.add(t.accept)
            if t.reject != "_":
                t.states.add(t.reject)

            for line in lines:
                p, a, b, dir, q = line.strip().split(",")
                if (p, a) not in t.T:
                    t.T[(p, a)] = set()
                t.T[(p, a)].add((b, dir, q))
                t.states.add(p)
                t.states.add(q)
                if a not in t.auxalpha:
                    raise Exception("Transição com letra não declarada.")

        return t

    def run(self, tape: str, steps: int = -1) -> runresult:
        """
        Executa a máquina com a fita de entrada dada pela string tape.
        O parâmetro opcional steps estipula a quantidade máxima de passos da execução.
        """

        r = runresult(False, 0, [], self.initial)

        tape = deque(tape)
        pos = 0

        r.computation.append((copy.deepcopy(tape), pos, r.state))

        print(r.state)
        print("".join(tape))
        print("^")

        while r.steps != steps:  # Se steps não for definido, aqui sempre será True
            s = self.T[(r.state, tape[pos])]
            if not s:
                raise Exception("Transição não definida.")
            t, dir, q = random.choice(list(s))
            tape[pos] = t
            r.state = q
            if dir == "D":
                pos += 1
                if pos == len(tape):
                    tape.append("_")
            if dir == "E":
                pos -= 1
                if pos < 0:
                    tape.appendleft("_")
                    pos = 0
            r.steps += 1
            r.computation.append((copy.deepcopy(tape), pos, r.state))
            if r.state == self.accept:
                r.accept = True
                return r
            if r.state == self.reject:
                r.accept = False
                return r

            print(r.state)
            print("".join(tape))
            print(" " * pos + "^")


if __name__ == "__main__":
    spec = r"C:\Users\Diretoria DC\Downloads\div3.dfa"
    A = dfa.fromfile(spec)
    A.printmachine()
    spec = r"C:\Users\Diretoria DC\Downloads\div2.dfa"
    B = dfa.fromfile(spec)
    B.printmachine()

    C = dfa.product(A, B)
    C.printmachine()

    D = dfa("ab")
    D.addstate("q1")
    D.addstate("q2")
    D.addstate("q3")
    D.setinitial("q1")
    D.setfinal("q2")

    D.addtransition("q1", "q1", "b")
    D.addtransition("q1", "q2", "a")
    D.addtransition("q2", "q3", "a")
    D.addtransition("q2", "q3", "b")
    D.addtransition("q3", "q2", "a")
    D.addtransition("q3", "q1", "b")

    D.printmachine()

    q = D.act("aabb")
    print(q)

    e = simpleRE("a*+(bb+aba*)ab")
    e.printtree()

"""
    mytm = r"C:\ Users\Diretoria DC\Downloads\ss.tm"
    t = ntm.fromfile(mytm)

    fita = "1010|110|1011001|111101|10|100010|10010|1100100|1001|1001001"

    r = t.run(fita)

    print(r.computation)
"""


"""

    mytm = r"C:\ Users\Diretoria DC\Downloads\akbk.tm"

    # mytm = r"C:\ Users\Diretoria DC\Downloads\primitiva.tm"

    t = tm.fromfile(mytm)

    print(t.T)

    tapes = []
    for k in range(1, 51):
        tapes.append("a" * k + "b" * k)

    x, y = t.complexity(tapes, lambda x: x**2 / 2, label=r"$\frac{x^2}{2}$")

"""

"""
    fita = "aaaaaabbbbbb"

    r = t.run(fita)

    if r.state == t.accept:
        print("Cadeia aceita.")
    if r.state == t.reject:
        print("Cadeia rejeitada")

    print(r.steps)
"""
