{% extends "autosummary/module.rst" %}
{% block title -%}

{{ ("``" ~ fullname ~ "``") | underline('=')}}

{%- endblock %}
