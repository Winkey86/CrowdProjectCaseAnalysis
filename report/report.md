# Краткая сводка по Crowd-проекту

## Ключевые значения

- Ассайнменты: 21 437
- Item-строки: 203 963
- Исполнители: 2 258
- APPROVED: 18 579
- EXPIRED: 2 815
- SKIPPED: 43
- Уникальные контрольные задания: 12 698
- Контроли с достаточным числом голосов (`n_valid_votes >= 4`): 1 405
- Проблемные контроли: 412
- Сложные контроли: 289
- Неоднозначные контроли: 123

## Метрики проекта

| metric | formula | value | why_it_matters |
| --- | --- | --- | --- |
| completion_rate | APPROVED assignments / all assignments | 86.7% | Операционная стабильность: доля страниц, завершённых и одобренных. |
| unfinished_assignment_share | (EXPIRED assignments + SKIPPED assignments) / all assignments | 13.3% | Потери потока: доля EXPIRED + SKIPPED среди всех страниц. |
| control_accuracy (основная метрика качества проекта) | correct worker-control votes / valid worker-control votes | 71.9% | Качество разметки: точность исполнителей на контрольных заданиях. |
| problem_control_share | (difficult controls + ambiguous controls) / controls with n_valid_votes >= 4 | 29.3% | Качество контрольного пула: доля сложных/неоднозначных контролей среди контролей с n>=4. |
| suspicious_worker_share | workers with any risk flag / all workers | 28.9% | Риск по исполнителям: доля работников с признаками низкого качества или неустойчивого поведения. |

## Топ проблемных контролей

| url | gold_answer | n_valid_votes | correct_share | incorrect_share | flag |
| --- | --- | --- | --- | --- | --- |
| https://gazballon-nizhniy-tagil.ru | no | 8 | 0.000 | 1.000 | difficult |
| https://efsmoscow.ru | no | 7 | 0.000 | 1.000 | difficult |
| https://cheese-box.ru | yes | 6 | 0.000 | 1.000 | difficult |
| https://faridkamal.pro | not_working | 6 | 0.000 | 1.000 | difficult |
| https://filternn.ru | no | 6 | 0.000 | 1.000 | difficult |
| https://food-service.ru | yes | 6 | 0.000 | 1.000 | difficult |
| https://grocenberg.store | no | 6 | 0.000 | 1.000 | difficult |
| https://ishop72.ru | no | 6 | 0.000 | 1.000 | difficult |
| https://les-wm.ru | no | 6 | 0.000 | 1.000 | difficult |
| https://m-open.ru | no | 6 | 0.000 | 1.000 | difficult |

## Топ подозрительных исполнителей

| worker_id | approved_count | items_done | controls_done | control_accuracy | answer_dominance_share | suspicious_reasons |
| --- | --- | --- | --- | --- | --- | --- |
| 9ebe1eac-033d-494c-870f-94a9d666134b | 2 | 20 | 4 | 0.000 | 0.950 | low_control_accuracy,answer_dominance,fast_low_quality |
| e64aa1b3-d75e-42a9-95ce-0afdcf926e50 | 2 | 20 | 4 | 0.000 | 0.950 | low_control_accuracy,answer_dominance,fast_low_quality |
| ce4aa68e-b693-46f4-b477-e996bef4f0bd | 3 | 30 | 3 | 0.000 | 0.933 | answer_dominance |
| 7bd77f54-800e-4c5a-8f50-f311ed17c2e6 | 2 | 20 | 4 | 0.000 | 0.900 | low_control_accuracy,answer_dominance |
| c81a4911-feda-41e7-a1f6-44173f0ec99b | 2 | 20 | 4 | 0.000 | 0.900 | low_control_accuracy,answer_dominance,fast_low_quality |
| e7e9e331-a9e1-4a7a-b37f-fd1279837283 | 2 | 20 | 2 | 0.000 | 0.900 | answer_dominance |
| 5ff603d8-25bb-4b85-9125-887c9c252c62 | 1 | 10 | 2 | 0.000 | 0.800 | unfinished_behavior |
| 3bba6a3e-6116-4cdb-8dc5-a8d138445597 | 2 | 20 | 4 | 0.000 | 0.750 | low_control_accuracy,fast_low_quality,unfinished_behavior |
| ad844971-2f67-4c61-a172-04187a47618f | 2 | 20 | 4 | 0.000 | 0.700 | low_control_accuracy |
| b1877f4f-f17b-44a0-b8b6-ebc2e9c7266f | 4 | 40 | 4 | 0.000 | 0.675 | low_control_accuracy |

## Топ спорных обычных URL

| url | shown_count | confidence | final_label | yes_count | no_count | not_working_count |
| --- | --- | --- | --- | --- | --- | --- |
| https://01-kras.ru | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://100-del.ru | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://1000.com.ru | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://14element.ru | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://153рыбы.рф | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://1688seller.ru | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://1bint.ru | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://1c-prices.ru | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://4fcompany.ru | 3 | 0.333 | tie | 1 | 1 | 1 |
| https://59zimaleto.ru | 3 | 0.333 | tie | 1 | 1 | 1 |

## Артефакты

- `dataset/items_long.csv` — основной long-датасет.
- `control_task_metrics/control_task_metrics.csv` — все контрольные задания с метриками.
- `control_task_metrics/problem_controls.csv` — difficult/ambiguous контроли.
- `report/project_metrics.csv` — ключевые метрики проекта.
- `report/top_suspicious_workers.csv` — исполнители с риск-флагами.
- `report/top_controversial_urls.csv` — URL с низкой согласованностью голосов.
- `report/control_correct_share_histogram.png` — график распределения `correct_share` по контролям.
