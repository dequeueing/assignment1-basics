def apply_discount(price, discount_rate):
    # 模拟一个复杂的折扣计算逻辑
    final_price = price * (1 - discount_rate)
    return final_price

def calculate_total_bill(items):
    total = 0
    for item in items:
        name = item['name']
        price = item['price']
        
        # 假设：如果商品价格超过 100，就给 20% 的折扣
        if price > 100:
            discount = 0.2
        else:
            discount = 0 # 没有折扣
            
        current_price = apply_discount(price, discount)
        total += current_price
        
    return total

# 购物清单
shopping_cart = [
    {'name': '键盘', 'price': 150},
    {'name': '鼠标垫', 'price': 50},
    {'name': '显示器', 'price': 300}
]

result = calculate_total_bill(shopping_cart)
print(f"最终总账单应该是: {result}")
# 预期结果：(150*0.8) + 50 + (300*0.8) = 120 + 50 + 240 = 410
# 实际运行你会发现：结果可能不对（或者你想验证每一环节）