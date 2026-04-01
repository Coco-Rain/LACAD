import cadquery as cq

result = (
cq.Workplane("XY")
.circle(12).extrude(6)
.faces(">Z").workplane()
.polygon(5, 8).extrude(4)
.edges("|Z").chamfer(1.5)
.union(
cq.Workplane("XZ")
.transformed(offset=(0, 4, 3))
.circle(5).extrude(8, both=True)
)
)
cq.exporters.export(result, 'GT.stl')