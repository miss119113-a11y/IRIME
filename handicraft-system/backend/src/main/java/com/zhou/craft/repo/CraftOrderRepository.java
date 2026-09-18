package com.zhou.craft.repo;
import com.zhou.craft.model.CraftOrder;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
public interface CraftOrderRepository extends JpaRepository<CraftOrder,Long>{
    List<CraftOrder> findAllByOrderByCreatedAtDesc();
    List<CraftOrder> findByCustomerNameOrderByCreatedAtDesc(String customerName);
}
