#Набор входных данных

Представим фон и 10 предметов для поставленной задачи

##Выбранный фон:
![background](https://user-images.githubusercontent.com/72349827/190858056-485741ff-ea9c-429c-bf39-a7c4194a3a61.jpg)

##Выбранные предметы
1. Ручка
![pen](https://user-images.githubusercontent.com/72349827/190858077-7a29ec3f-c42f-4588-8d1a-d14ec33583c4.jpg)
2. Резинка для волос
![scrunchy](https://user-images.githubusercontent.com/72349827/190858086-1c0e18af-866c-4632-b849-161d79b95223.jpg)
3. Ножницы
![scissors](https://user-images.githubusercontent.com/72349827/190858081-38bca7b9-7035-41f0-a2fe-0bca5bdcecce.jpg)
4. Ключ
![key](https://user-images.githubusercontent.com/72349827/190858073-3dce7b1c-1142-40c2-a333-84d226250cee.jpg)
5. Значок
![badge](https://user-images.githubusercontent.com/72349827/190858063-f041392c-efbe-4b43-9639-40e21b8f5445.jpg)
6. Кольцо
![ring](https://user-images.githubusercontent.com/72349827/190858080-dc7a4b6c-3984-49ed-960e-060a13105b52.jpg)
7. Чехол для наушников
![case](https://user-images.githubusercontent.com/72349827/190858065-7d632105-9b94-4554-b707-69b7fd1c2fbb.jpg)
8. Помада
![lipstick](https://user-images.githubusercontent.com/72349827/190858074-8ce1002f-2147-44e8-98cb-1696bbe22736.jpg)
9. Тени для век
![eyeshadow](https://user-images.githubusercontent.com/72349827/190858067-15ca5d7e-76e0-4d48-a8ad-b76a02a284c6.jpg)
10. Канцелярский зажим
![paperclip](https://user-images.githubusercontent.com/72349827/190858075-4650c42a-19ec-471c-91ff-4d3cd651cb1f.jpg)

Будем рассматривать случаи покрывающие как можно больше возможных типов конфигураций предметов и видов многоугольника.

#1. Простой случай соответствия размера
Рассмотрим самый очевидный и простейший случай. На вход подаётся большой четырёхугольник и чехол
![1](https://user-images.githubusercontent.com/72349827/190858401-e8c1c983-27f3-4200-a3d8-41ac9387958b.jpg)
Выход:
YES

#2. Случай несоответствия размера c одним предметом
На вход подается треугольник и ножницы, которые по размеру не помещаются в фигуру
![2](https://user-images.githubusercontent.com/72349827/190859182-c6736969-1a1a-4677-992d-dd3daab71267.jpg)
Выход:
NO
![3](https://user-images.githubusercontent.com/72349827/190859207-5c91eb10-2ace-4e7e-96a6-bb89060bf572.jpg)
Выход:
NO

#3. Случай несоответствия размера c двумя предметами
На вход подается треугольник, помада и ключ, которые по размеру не помещаются в фигуру
![12](https://user-images.githubusercontent.com/72349827/190859519-a9919a91-2b54-44c7-a30e-ad4e7b70d9ff.jpg)
Выход:
NO

#4. Предельный случай с одним предметом
На входе имеем прямоугольник, который может вплотную поместить в себя ручку 
![4](https://user-images.githubusercontent.com/72349827/190859214-6ea2dd9b-0976-4300-a799-9f74515fe534.jpg)
Выход:
YES

#5. Предельный случай с несколькими предметами
На входе имеем четырехугольник и несколько предметов. Рассмотрим случаи, когда в зависимостиот перемещения предметов получаем разные ответы
![5](https://user-images.githubusercontent.com/72349827/190859263-286f90cd-ccc3-4037-b26b-087844c60c07.jpg)
Выход:
YES
![6](https://user-images.githubusercontent.com/72349827/190859265-f305f8f1-558e-4365-a2fe-7fecdf08510f.jpg)
Выход:
NO
![7](https://user-images.githubusercontent.com/72349827/190859268-ebfee650-8ba4-4590-8890-dcf575e43c2e.jpg)
Выход:
YES

#6.Случай неподходящих многоугольников
На входе получаем различные геометрические фигуры неудовлетворяющие условиям задачи
![8](https://user-images.githubusercontent.com/72349827/190859414-87218125-d465-420f-8830-45596728307e.jpg)
Выход:
NO
![9](https://user-images.githubusercontent.com/72349827/190861007-071fe96c-2c11-4d4e-a454-50cfd616b98e.jpg)
Выход:
NO
![10](https://user-images.githubusercontent.com/72349827/190861035-6de179e1-f2f9-472b-aa57-77cea878b43f.jpg)
Выход:
NO
![11](https://user-images.githubusercontent.com/72349827/190861057-78715ebe-6ea1-4ff9-8460-7bd6146f5d42.jpg)
Выход:
NO

#7. Случай с большим количеством предметов
На входе имеем прямоугольник и все множество допустимых предметов (в количестве 10 штук).
![13](https://user-images.githubusercontent.com/72349827/190861077-39e931b6-502c-44c4-94af-11635a0c1d22.jpg)
Выход:
NO
![14](https://user-images.githubusercontent.com/72349827/190861103-4a0dc4b1-0066-4148-a22e-604150cc7c61.jpg)
Выход:
YES

#8. Случай с отсутствием предметов
На вход поступает многоугольник без предметов
![15](https://user-images.githubusercontent.com/72349827/190859751-99f3469f-550f-4e9d-a8ed-88ba4e77c4ed.jpg)
Выход:
YES

#9. Другие случаи
Рассмотрим также обычные случаи работы алгоритма
![16](https://user-images.githubusercontent.com/72349827/190861239-7ba17e66-72e6-463e-8607-d9e2fec3f9a9.jpg)
Выход:
YES
![17](https://user-images.githubusercontent.com/72349827/190861240-b79f3b70-9f96-41da-9725-7d4e5ba28594.jpg)
Выход:
YES
![18](https://user-images.githubusercontent.com/72349827/190861244-f3f9b1d4-cd3f-4dab-b1dc-5be90c2e14ac.jpg)
Выход:
YES
![19](https://user-images.githubusercontent.com/72349827/190861247-885095ff-0b09-463c-a93b-55c9cfbda0b7.jpg)
Выход:
NO
![20](https://user-images.githubusercontent.com/72349827/190861248-6212524e-b7a7-42d7-92d6-c5f910fa1566.jpg)
Выход:
YES
![21](https://user-images.githubusercontent.com/72349827/190861249-ac66884f-57b8-486c-aafb-e1749aa51c21.jpg)
Выход:
YES
![22](https://user-images.githubusercontent.com/72349827/190861250-1ad7dd0b-9017-40c8-847d-5d390a006341.jpg)
Выход:
YES
![23](https://user-images.githubusercontent.com/72349827/190861251-9dd90559-ec45-4b85-8226-99e5db23522f.jpg)
Выход:
NO
![24](https://user-images.githubusercontent.com/72349827/190861253-97bd1077-9e18-413b-a1a0-5c42895cdeec.jpg)
Выход:
YES
![25](https://user-images.githubusercontent.com/72349827/190861255-f72b9322-22c2-40ed-a632-1abe2fb95a81.jpg)
Выход:
YES
![26](https://user-images.githubusercontent.com/72349827/190861257-66e5d427-8f34-4555-b89a-131897d51cdb.jpg)
Выход:
NO
![27](https://user-images.githubusercontent.com/72349827/190861258-23d1f354-8d85-46c1-8812-c949e10c8849.jpg)
Выход:
NO
