import os
import sys
from dataclasses import dataclass

from django.conf import settings
from django.core.wsgi import get_wsgi_application
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import path
from django.utils.crypto import get_random_string

settings.configure(
    DEBUG=(os.environ.get("DEBUG", "") == "1"),
    ALLOWED_HOSTS=["*"],
    ROOT_URLCONF=__name__,
    SECRET_KEY=get_random_string(50),
    MIDDLEWARE=["django.middleware.common.CommonMiddleware"],
)


@dataclass
class Character:
    name: str
    age: int

    def as_dict(self, id_):
        return {
            "id": id_,
            "name": self.name,
            "age": self.age,
        }


characters = {
    1: Character("Rick Sanchez", 70),
    2: Character("Morty Smith", 14),
}


def index(request):
    return HttpResponseRedirect("/characters/")


def characters_list(request):
    return JsonResponse(
        {"data": [character.as_dict(id_) for id_, character in characters.items()]}
    )


def characters_detail(request, character_id):
    try:
        character = characters[character_id]
    except KeyError:
        return JsonResponse(
            status=404,
            data={"error": f"Character with id {character_id!r} does not exist."},
        )
    return JsonResponse({"data": character.as_dict(character_id)})


urlpatterns = [
    path("", index),
    path("characters/", characters_list),
    path("characters/<int:character_id>/", characters_detail),
]

app = get_wsgi_application()

import cinderx.jit
cinderx.jit.auto()

import pprint

def print_deopt_stats() -> None:
    pprint.pprint(cinderx.jit.get_and_clear_runtime_stats())

import atexit
atexit.register(print_deopt_stats)
