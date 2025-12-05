from src.logic.utils.helpers import get_lookup


lookup = {
    "Loja": "Campina Grande",
    "Area Total": "135,65",
    "Caixas": "6",
    "Qtd Prateleiras": "40",
    "Escritorio": "VERDADEIRO",
    "Copa": "VERDADEIRO",
    "Espaco Evento": "VERDADEIRO",
}
print(get_lookup(lookup, "Qtd Caixas"))
print(get_lookup(lookup, "Caixas"))
print(get_lookup(lookup, "Esp Conv"))
