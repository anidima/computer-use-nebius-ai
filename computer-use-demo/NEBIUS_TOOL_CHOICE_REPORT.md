# Отчет о тестировании tool_choice в Nebius - Сравнение 4 моделей

## 🔬 Тестируемые модели:
1. **mistralai/Mistral-Small-3.1-24B-Instruct-2503**
2. **google/gemma-3-27b-it**
3. **Qwen/Qwen2-VL-72B-Instruct**
4. **Qwen/Qwen2.5-VL-72B-Instruct**

## 📋 Результаты тестирования по моделям:

### 1. � mistralai/Mistral-Small-3.1-24B-Instruct-2503 (0/5 тестов прошли)

| tool_choice | Результат | Ошибка |
|-------------|-----------|---------|
| `None` | ❌ | This model does not support auto tool |
| Без параметра | ❌ | This model does not support auto tool |
| `"auto"` | ❌ | This model does not support auto tool |
| `"required"` | ❌ | Chat template error (transformers v4.44) |
| Конкретный инструмент | ❌ | Chat template error (transformers v4.44) |

### 2. � google/gemma-3-27b-it (0/5 тестов прошли)

| tool_choice | Результат | Ошибка |
|-------------|-----------|---------|
| `None` | ❌ | This model does not support auto tool |
| Без параметра | ❌ | This model does not support auto tool |
| `"auto"` | ❌ | This model does not support auto tool |
| `"required"` | ❌ | JSON schema not supported by xgrammar |
| Конкретный инструмент | ❌ | JSON schema not supported by xgrammar |

### 3. 🟡 Qwen/Qwen2-VL-72B-Instruct (2/6 тестов прошли)

| tool_choice | Результат | Ошибка/Результат |
|-------------|-----------|---------|
| `None` | ❌ | This model does not support auto tool |
| Без параметра | ❌ | This model does not support auto tool |
| `"auto"` | ❌ | This model does not support auto tool |
| `"required"` | ❌ | Must be named tool, "auto", or "none" |
| `"none"` | ✅ | **РАБОТАЕТ!** Не использует инструменты |
| Конкретный инструмент | ✅ | **РАБОТАЕТ!** Вызвал computer tool |

### 4. 🟢 Qwen/Qwen2.5-VL-72B-Instruct (3/6 тестов прошли)

| tool_choice | Результат | Ошибка/Результат |
|-------------|-----------|---------|
| `None` | ❌ | This model does not support auto tool |
| Без параметра | ❌ | This model does not support auto tool |
| `"auto"` | ❌ | This model does not support auto tool |
| `"required"` | ✅ | **РАБОТАЕТ!** Вызвал str_replace_editor |
| `"none"` | ✅ | **РАБОТАЕТ!** Не использует инструменты |
| Конкретный инструмент | ✅ | **РАБОТАЕТ!** Вызвал computer tool |

## 🎯 Ключевые выводы:

### ❌ НЕ РАБОТАЕТ ни в одной модели:
- `tool_choice=None` (Python None)
- Отсутствие `tool_choice` при наличии `tools`
- `tool_choice="auto"`

**Все модели возвращают одинаковую ошибку:**
```
This model does not support auto tool, please use tool_choice.
```

### ✅ РАБОТАЕТ частично:

1. **Qwen/Qwen2.5-VL-72B-Instruct** - ЛУЧШАЯ поддержка (3/6):
   - ✅ `tool_choice="required"` 
   - ✅ `tool_choice="none"`
   - ✅ `tool_choice={"type": "function", "function": {"name": "tool_name"}}`

2. **Qwen/Qwen2-VL-72B-Instruct** - Хорошая поддержка (2/6):
   - ✅ `tool_choice="none"`
   - ✅ `tool_choice={"type": "function", "function": {"name": "tool_name"}}`
   - ❌ `tool_choice="required"` (валидационная ошибка)

3. **mistralai/Mistral-Small** и **google/gemma-3** - НЕТ поддержки инструментов

### 🔧 Специфичные ошибки по моделям:

- **Mistral**: Проблемы с chat template (transformers v4.44)
- **Gemma**: JSON schema incompatible с xgrammar
- **Qwen2-VL**: Не поддерживает `"required"` 
- **Qwen2.5-VL**: Наилучшая поддержка инструментов

## 🚨 КРИТИЧЕСКАЯ ПРОБЛЕМА для loop.py:

**Текущий код в loop.py строка 436:**
```python
tool_choice=None  # ❌ Это НЕ РАБОТАЕТ ни в одной модели!
```

## 💡 РЕКОМЕНДАЦИИ:

### Для кода в loop.py:

```python
# Вместо tool_choice=None используйте:
if provider == APIProvider.NEBIUS:
    if model in ["Qwen/Qwen2.5-VL-72B-Instruct"]:
        tool_choice = "required"  # Лучший вариант для Qwen2.5-VL
    elif model in ["Qwen/Qwen2-VL-72B-Instruct"]:
        # Укажите конкретный инструмент, например первый доступный
        tool_choice = {"type": "function", "function": {"name": first_tool_name}}
    else:
        # Для Mistral и Gemma - не используйте инструменты вообще
        tools = None
        tool_choice = None
else:
    tool_choice = None  # Для других провайдеров (Anthropic, etc.)
```

### Итоговый рейтинг моделей по поддержке инструментов:

1. 🥇 **Qwen/Qwen2.5-VL-72B-Instruct** - отличная поддержка (3/6)
2. 🥈 **Qwen/Qwen2-VL-72B-Instruct** - хорошая поддержка (2/6)  
3. 🥉 **mistralai/Mistral-Small-3.1-24B-Instruct-2503** - нет поддержки (0/6)
4. 🥉 **google/gemma-3-27b-it** - нет поддержки (0/6)

### 🔍 ВАЖНОЕ ОТКРЫТИЕ про `tool_choice="none"`:

Обе модели Qwen поддерживают `tool_choice="none"`, что позволяет:
- Передавать инструменты в запросе
- Но явно указать модели НЕ использовать их
- Это полезно для fallback сценариев

**Пример использования:**
```python
# Если хотите дать модели доступ к инструментам, но не принуждать их использовать
tool_choice = "none"  # Работает для Qwen моделей

# Если хотите принудить использование инструментов
tool_choice = "required"  # Работает только для Qwen2.5-VL

# Если хотите указать конкретный инструмент  
tool_choice = {"type": "function", "function": {"name": "computer"}}  # Работает для обеих Qwen
```
