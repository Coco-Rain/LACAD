import cadquery as cq

result = (
cq.Workplane("XY")
.cylinder(12, 6)
.union(
cq.Workplane("YZ")
.transformed(offset=(0, 0, 6))
.sphere(8)
)
.faces("<X").workplane()
.slot2D(10, 3, 0)
.cutThruAll()
)
cq.exporters.export(result, 'GT.stl')